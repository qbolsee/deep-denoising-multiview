import glob
import os
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import io

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "train_dir",
    "log_train/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/",
    """Directory where to write event logs""",
)
tf.app.flags.DEFINE_string(
    "dataset_train",
    "../dataset/training/patch_2d/train_*.tfrecord",
    """Location of the dataset files""",
)
tf.app.flags.DEFINE_string(
    "dataset_valid",
    "../dataset/training/patch_2d/valid_*.tfrecord",
    """Location of the dataset files""",
)
tf.app.flags.DEFINE_integer("train_n", 4, """Number of training datasets""")
tf.app.flags.DEFINE_integer("valid_n", 1, """Number of training datasets""")
tf.app.flags.DEFINE_integer(
    "dataset_size", 100000, """Number of examples per dataset file"""
)
tf.app.flags.DEFINE_integer("patch", 64, """Number of examples per dataset file""")

tf.app.flags.DEFINE_integer("batch_size", 128, """Batch size""")
tf.app.flags.DEFINE_integer("max_steps", 600000, """Max number of batches to run""")

tf.app.flags.DEFINE_float(
    "epochs_per_decay", 4, """How many epochs to run before 1 decay step"""
)
tf.app.flags.DEFINE_float("initial_lr", 0.0008, """Initial learning rate""")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.9, """Learning rate decay factor""")


SCALE_DEPTH = 9.375  # MAX 9.375 meters, 16MHz


def parse_img(tf_bytes, h, w, c):
    return tf.reshape(tf.decode_raw(tf_bytes, tf.float32), [h, w, c])


def parse_example(example_proto):
    h = FLAGS.patch
    w = FLAGS.patch
    features = {
        "depth_n": tf.FixedLenFeature((), tf.string, default_value=""),
        "depth_gt": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return {
        "depth_n": parse_img(parsed_features["depth_n"], h, w, 1),
        "depth_gt": parse_img(parsed_features["depth_gt"], h, w, 1),
    }


def dataset_get_next(training):
    path = FLAGS.dataset_train if training else FLAGS.dataset_valid
    filenames = []
    for f in glob.glob(path):
        filenames.append(f)
    if training:
        with tf.variable_scope("dataset_train") as scope:
            dataset_train = tf.data.TFRecordDataset(filenames)
            dataset_train = dataset_train.map(parse_example)
            dataset_train = dataset_train.shuffle(buffer_size=10000)
            dataset_train = dataset_train.batch(FLAGS.batch_size)
            dataset_train = dataset_train.repeat()

            iterator = dataset_train.make_initializable_iterator()
            next_entry = iterator.get_next()
    else:
        with tf.variable_scope("dataset_valid") as scope:
            dataset_test = tf.data.TFRecordDataset(filenames)
            dataset_test = dataset_test.map(parse_example)
            dataset_test = dataset_test.batch(FLAGS.batch_size)
            dataset_test = dataset_test.repeat()

            iterator = dataset_test.make_initializable_iterator()
            next_entry = iterator.get_next()

    return next_entry["depth_n"], next_entry["depth_gt"], iterator.initializer


def run_forward(img_in, flag_train):
    F_IN = 3
    F_HIDDEN = 3
    F_OUT = 3
    D = 12

    N_CONV_IN = 64
    N_CONV_HIDDEN = 64

    # padding = (F_IN // 2) + (D - 2) * (F_HIDDEN // 2) + (F_OUT // 2)

    # preprocess
    with tf.variable_scope("preprocess") as scope:
        mask = tf.cast(img_in > 0, tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=[1, 2], keep_dims=True)
        mask_sum = tf.maximum(mask_sum, 1)
        d_mean = tf.reduce_sum(img_in * mask / mask_sum, axis=[1, 2], keep_dims=True)
        d_std = tf.sqrt(
            tf.reduce_sum(
                tf.square((img_in - d_mean) * mask) / mask_sum,
                axis=[1, 2],
                keep_dims=True,
            )
        )
        d_scale = tf.maximum(d_std, tf.constant(0.00001, name="eps"))
        img_in_norm = tf.div(
            tf.subtract(img_in, d_mean) * mask, d_scale, name="img_in_pre"
        )

    # 64 -> 64 -> 64                                                        64 -> 64 -> 64
    #             32 -> 32 -> 32                                32 -> 32 -> 32
    #                         16 -> 16 -> 16        16 -> 16 -> 16
    #                                      8 -> 8 -> 8

    with tf.variable_scope("conv1") as scope:
        # x = tf.pad(img_in_norm, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]]))
        x = img_in_norm
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=F_IN, activation="relu", padding="same"
        )(x)
        x1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=F_IN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv2") as scope:
        x = tf.keras.layers.MaxPool2D()(x1)
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)
        x2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv3") as scope:
        x = tf.keras.layers.MaxPool2D()(x2)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)
        x3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv4") as scope:
        x = tf.keras.layers.MaxPool2D()(x3)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)
        x4 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv3_up") as scope:
        x = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, strides=(2, 2), padding="same", activation="relu"
        )(x4)
        x = tf.keras.layers.Concatenate()([x, x3])
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)
        x3_up = tf.keras.layers.Conv2D(
            filters=64, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv2_up") as scope:
        x = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=4, strides=(2, 2), padding="same", activation="relu"
        )(x3_up)
        x = tf.keras.layers.Concatenate()([x, x2])
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)
        x2_up = tf.keras.layers.Conv2D(
            filters=32, kernel_size=F_HIDDEN, activation="relu", padding="same"
        )(x)

    with tf.variable_scope("conv1_up") as scope:
        x = tf.keras.layers.Conv2DTranspose(
            filters=16, kernel_size=4, strides=(2, 2), padding="same", activation="relu"
        )(x2_up)
        x = tf.keras.layers.Concatenate()([x, x1])
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=F_OUT, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=F_OUT, activation="relu", padding="same"
        )(x)
        x_out = tf.keras.layers.Conv2D(filters=1, kernel_size=F_OUT, padding="same")(x)

    with tf.variable_scope("output") as scope:
        x = tf.multiply(x_out, d_scale)
        img_out = tf.subtract(img_in[:, :, :, 0:1], x)

    return img_out


def run_loss(img_out, img_gt, scope_name="loss_compute"):
    with tf.variable_scope(scope_name) as scope:
        mask = tf.cast(img_gt > 0, tf.float32)
        diff = tf.subtract(img_out, img_gt)
        diff_square = tf.square(diff) * mask

        # x_grad = (img_out[:, 1:, 1:] - img_out[:, 1:, 0:-1])
        # y_grad = (img_out[:, 1:, 1:] - img_out[:, 0:-1, 1:])
        mask_sum = tf.reduce_sum(mask, axis=[1, 2], keep_dims=True)
        # loss_grad = 0.1 * tf.reduce_mean(x_grad*x_grad + y_grad*y_grad)
        # loss_val = tf.reduce_sum(diff_square/mask_sum, name="mean_square") + loss_grad
        loss_val = tf.reduce_sum(diff_square / mask_sum, name="mean_square")
    if scope_name == "loss_compute":
        tf.summary.scalar("loss train", loss_val, collections=[tf.GraphKeys.SUMMARY_OP])
    return loss_val


def run_psnr(img_in, img_out, img_ref, scope_name="psnr_compute"):
    with tf.variable_scope(scope_name) as scope:
        mask = tf.cast(img_ref > 0, tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=[1, 2], keep_dims=True)
        diff_square_in = tf.square(tf.subtract(img_in[:, :, :, 0:1], img_ref)) * mask
        diff_square_out = tf.square(tf.subtract(img_out, img_ref)) * mask
        mse_in = tf.reduce_sum(diff_square_in / mask_sum)
        mse_out = tf.reduce_sum(diff_square_out / mask_sum)
        psnr_in = 10.0 * tf.log(SCALE_DEPTH * SCALE_DEPTH / mse_in) / tf.log(10.0)
        psnr_out = 10.0 * tf.log(SCALE_DEPTH * SCALE_DEPTH / mse_out) / tf.log(10.0)
        psnr_diff = tf.subtract(psnr_out, psnr_in)
    if scope_name == "psnr_compute":
        tf.summary.scalar(
            "PSNR train", psnr_diff, collections=[tf.GraphKeys.SUMMARY_OP]
        )
    return psnr_diff


def run_training(total_loss, global_step):
    # Variables that affect learning rate.
    num_examples_per_epoch = FLAGS.train_n * FLAGS.dataset_size
    num_batches_per_epoch = num_examples_per_epoch / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        FLAGS.initial_lr,
        global_step,
        decay_steps,
        FLAGS.lr_decay_factor,
        staircase=True,
    )
    lr_summary = tf.summary.scalar(
        "learning rate", lr, collections=[tf.GraphKeys.SUMMARY_OP]
    )

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(
        grads, global_step=global_step
    )  # increments global_step

    return apply_gradient_op, lr_summary


def main_train():
    with tf.Graph().as_default():
        global_step = tf.Variable(
            0, trainable=False, name="global_step"
        )  # steps counter

        # input data
        img_in_train, img_gt_train, init_train = dataset_get_next(training=True)
        img_in_test, img_gt_test, init_test = dataset_get_next(training=False)

        # input placeholder
        with tf.variable_scope("input") as scope:
            img_in = tf.placeholder(
                name="img_in", dtype=tf.float32, shape=(None, None, None, 1)
            )
            img_gt = tf.placeholder(
                name="img_gt", dtype=tf.float32, shape=(None, None, None, 1)
            )
            flag_train = tf.placeholder(tf.bool, shape=(), name="flag_train")

        # forward pass
        img_out = run_forward(img_in, flag_train)
        # loss function
        total_loss = run_loss(img_out, img_gt, "loss_compute")
        total_loss_test = run_loss(img_out, img_gt, "loss_compute_test")
        # PSNR metric
        total_psnr = run_psnr(img_in, img_out, img_gt, "psnr_compute")
        total_psnr_test = run_psnr(img_in, img_out, img_gt, "psnr_compute_test")
        # backpropagation
        train_op = run_training(total_loss, global_step)

        # init session
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            init_train,
            init_test,
        )

        sess = tf.Session()
        sess.run(init_op)

        # summaries
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARY_OP))
        summary_test_op = tf.summary.merge(
            [
                tf.summary.scalar("test loss", total_loss_test),
                tf.summary.scalar("PSNR test", total_psnr_test),
            ]
        )
        # summary_test_op = tf.summary.merge([tf.summary.scalar("test loss", total_loss_test)])
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"[PARAMETERS={n}]")

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            img_in_load, img_gt_load = sess.run([img_in_train, img_gt_train])
            _, loss_value, psnr_value = sess.run(
                [train_op, total_loss, total_psnr],
                feed_dict={
                    "input/img_in:0": img_in_load,
                    "input/img_gt:0": img_gt_load,
                    "input/flag_train:0": True,
                },
            )
            duration = time.time() - start_time
            assert not np.isnan(loss_value), "Model diverged with loss = NaN"

            if step % 10 == 0:
                summary_str = sess.run(
                    summary_op,
                    feed_dict={
                        "input/img_in:0": img_in_load,
                        "input/img_gt:0": img_gt_load,
                        "input/flag_train:0": False,
                    },
                )
                summary_writer.add_summary(summary_str, step)
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = "{}: step {:d}, loss = {:9f}, ({:.1f} examples/sec; {:.3f} sec/batch)"
                print(
                    format_str.format(
                        datetime.now(),
                        step,
                        loss_value,
                        examples_per_sec,
                        sec_per_batch,
                    )
                )

            if step > 0 and step % 100 == 0:
                img_in_load, img_gt_load = sess.run([img_in_test, img_gt_test])
                summary_str, loss_value, psnr_value = sess.run(
                    [summary_test_op, total_loss_test, total_psnr_test],
                    feed_dict={
                        "input/img_in:0": img_in_load,
                        "input/img_gt:0": img_gt_load,
                        "input/flag_train:0": False,
                    },
                )
                summary_writer.add_summary(summary_str, step)
                print("test loss: {:9f}".format(loss_value))

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

        sess.close()


if __name__ == "__main__":
    main_train()
