import glob
import math
import os
import random
import sys
import time
from datetime import datetime
from operator import pos

import numpy as np
import tensorflow as tf

# BASE_DIR = os.path.dirname(__file__)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "train_dir",
    "log_train2/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/",
    """Directory where to write event logs""",
)
tf.app.flags.DEFINE_string(
    "dataset_train",
    "../dataset/training/patch_3d_all/train_*.tfrecord",
    """Location of the dataset files""",
)
tf.app.flags.DEFINE_string(
    "dataset_valid",
    "../dataset/training/patch_3d_all/valid_*.tfrecord",
    """Location of the dataset files""",
)
tf.app.flags.DEFINE_integer("train_n", 8, """Number of training datasets""")
tf.app.flags.DEFINE_integer("valid_n", 1, """Number of training datasets""")
tf.app.flags.DEFINE_integer(
    "dataset_size", 100000, """Number of examples per dataset file"""
)
tf.app.flags.DEFINE_integer("n_point", 256, """Number of examples per dataset file""")

tf.app.flags.DEFINE_integer("batch_size", 128, """Batch size""")

tf.app.flags.DEFINE_integer("max_steps", 1000000, """Max number of batches to run""")
tf.app.flags.DEFINE_float(
    "epochs_per_decay", 30, """How many epochs to run before 1 decay step"""
)
tf.app.flags.DEFINE_float("initial_lr", 0.0002, """Initial learning rate""")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.95, """Learning rate decay factor""")


tf.app.flags.DEFINE_bool("use_abs", False, """Train on the absolute value of dz""")
tf.app.flags.DEFINE_string("method", "pointnet", """Architecture option""")


def parse_pcl(tf_bytes, n, n_feat):
    return tf.reshape(tf.decode_raw(tf_bytes, tf.float32), [n, n_feat])


def parse_vector(tf_bytes, n):
    return tf.reshape(tf.decode_raw(tf_bytes, tf.float32), [n])


def parse_float(tf_bytes):
    return tf.reshape(tf.decode_raw(tf_bytes, tf.float32), ())


def parse_example(example_proto):
    features = {
        "xyz": tf.FixedLenFeature((), tf.string, default_value=""),
        "xyz_n": tf.FixedLenFeature((), tf.string, default_value=""),
        "ray": tf.FixedLenFeature((), tf.string, default_value=""),
        "dz": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    xyz = parse_pcl(parsed_features["xyz"], FLAGS.n_point, 3)
    xyz_n = parse_pcl(parsed_features["xyz_n"], FLAGS.n_point, 3)
    ray = parse_pcl(parsed_features["ray"], FLAGS.n_point, 3)
    dz_gt = parse_vector(parsed_features["dz"], FLAGS.n_point)

    return xyz, xyz_n, ray, dz_gt


def get_dataset(path):
    with tf.variable_scope("dataset"):
        filenames = []
        for f in glob.glob(path):
            filenames.append(f)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_example)
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        xyz, xyz_n, ray, dz_gt = iterator.get_next()
        data_in = tf.concat((xyz, xyz_n, ray), axis=-1)
    return data_in, dz_gt, iterator.initializer


def run_forward(data_in, is_training, bn_decay=0.95, use_bn=False, method="pointnet"):
    batch_size = data_in.get_shape()[0].value
    num_point = data_in.get_shape()[1].value

    with tf.variable_scope("preprocess") as scope:
        point_cloud = data_in[:, :, :3]
        point_cloud_n = data_in[:, :, 3:6]
        point_ray = data_in[:, :, 6:]

        eps = tf.constant(0.0001, name="eps")

        # mean = tf.reduce_mean(point_cloud, axis=1, keep_dims=True)
        # point_cloud -= mean
        # point_cloud_n -= mean

        var = tf.sqrt(tf.reduce_mean(tf.square(point_cloud_n), axis=1, keep_dims=True))
        # var = tf.sqrt(tf.reduce_mean(tf.square(point_cloud_n), axis=[1, 2], keep_dims=True))
        var_clipped = tf.maximum(var, eps)

        # z_scale = tf.squeeze(tf.slice(var_clipped, [0, 0, 2], [-1, -1, 1]), axis=[1, 2])
        z_scale = var[:, 0, 2]
        # z_scale = tf.squeeze(var_clipped, axis=[1, 2])

        point_cloud = tf.divide(point_cloud, var_clipped)
        point_cloud_n = tf.divide(point_cloud_n, var_clipped)

        # tf.summary.scalar("var", tf.reduce_mean(z_scale, axis=0), collections=[tf.GraphKeys.SUMMARY_OP])

        point_ray = tf.divide(point_ray, var_clipped)
        ray_length = tf.norm(point_ray, axis=2, keepdims=True)
        point_ray = tf.divide(point_ray, ray_length)

        # point_data = tf.concat((point_cloud, point_cloud_n, point_ray), axis=-1)
        point_data = tf.concat((point_cloud_n, point_ray), axis=-1)
        # n_length = tf.norm(point_cloud-point_cloud_n, axis=2, keepdims=True)
        # point_data = tf.concat((point_ray, point_cloud_n), axis=-1)

    with tf.variable_scope("CNN") as scope:
        with tf.variable_scope("branch1") as scope2:
            input_image = tf.expand_dims(point_data, -1)
            # Point functions (MLP implemented as conv2d)
            net = tf_util.conv2d(
                input_image,
                32,
                [1, point_data.shape[2]],
                padding="VALID",
                stride=[1, 1],
                bn=use_bn,
                is_training=is_training,
                scope="conv1",
                bn_decay=bn_decay,
            )
            # net = tf_util.max_pool2d(net, [2, 1], padding="VALID", scope="max1")

            net = tf_util.conv2d(
                net,
                32,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=use_bn,
                is_training=is_training,
                scope="conv2",
                bn_decay=bn_decay,
            )
            # net = tf_util.max_pool2d(net, [2, 1], padding="VALID", scope="max2")

            net = tf_util.conv2d(
                net,
                32,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=use_bn,
                is_training=is_training,
                scope="conv3",
                bn_decay=bn_decay,
            )
            # net = tf_util.max_pool2d(net, [2, 1], padding="VALID", scope="max3")

            net = tf_util.conv2d(
                net,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=use_bn,
                is_training=is_training,
                scope="conv4",
                bn_decay=bn_decay,
            )
            # net = tf_util.max_pool2d(net, [2, 1], padding="VALID", scope="max4")

            net = tf_util.conv2d(
                net,
                512,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=use_bn,
                is_training=is_training,
                scope="conv5",
                bn_decay=bn_decay,
            )

        # print(net.shape)
        # net = tf.multiply(net, xyz_input[:, :, tf.newaxis, 2:], "multiply")
        # print(net.shape)

        # Symmetric function: max pooling
        # net = tf_util.avg_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
        net = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool")
        # net2 = tf_util.avg_pool2d(net2, [num_point, 1], padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        # net2 = tf.reshape(net2, [batch_size, -1])
        # net = tf.concat((net, net2), axis=-1)

        """
        net = tf_util.fully_connected(net, 256, bn=use_bn, is_training=is_training,
                                        scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 128, bn=use_bn, is_training=is_training,
                                        scope='fc2', bn_decay=bn_decay)
        """
        net = tf_util.fully_connected(
            net, 128, bn=use_bn, is_training=is_training, scope="fc1", bn_decay=bn_decay
        )
        net = tf_util.fully_connected(
            net, 64, bn=use_bn, is_training=is_training, scope="fc2", bn_decay=bn_decay
        )
        net = tf_util.fully_connected(
            net, 64, bn=use_bn, is_training=is_training, scope="fc3", bn_decay=bn_decay
        )

        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        # net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope="fc4")

        z_out = tf.squeeze(net, axis=-1)

    with tf.variable_scope("postprocess"):
        z_out = tf.multiply(z_out, z_scale, name="output")
        # z_out += mean[:, 0, 2]

    return z_out


def run_loss(dz_out, dz_gt):
    with tf.variable_scope("loss_compute") as scope:
        diff_square = tf.square(dz_gt - dz_out, name="diff_square")
        mse_before = tf.reduce_mean(tf.square(dz_gt), name="mse_before")

        mse = tf.reduce_mean(diff_square, name="mse")
        rmse = tf.sqrt(mse)

        psnr_before = 4.343 * tf.log(9.375 * 9.375 / mse_before)
        psnr = 4.343 * tf.log(9.375 * 9.375 / mse)
        psnr_delta = psnr - psnr_before

        tf.summary.scalar("mse", mse, collections=[tf.GraphKeys.SUMMARY_OP])
        tf.summary.scalar("rmse", rmse, collections=[tf.GraphKeys.SUMMARY_OP])
        tf.summary.scalar(
            "input rmse", tf.sqrt(mse_before), collections=[tf.GraphKeys.SUMMARY_OP]
        )

        tf.summary.scalar(
            "delta_PSNR", psnr_delta, collections=[tf.GraphKeys.SUMMARY_OP]
        )
    return mse, rmse


def run_valid(dz_out, dz_gt):
    with tf.variable_scope("valid_compute") as scope:
        diff_square = tf.square(dz_gt - dz_out, name="diff_square")
        mse_before = tf.reduce_mean(tf.square(dz_gt), name="mse_before")

        mse = tf.reduce_mean(diff_square, name="mse")
        rmse = tf.sqrt(mse)

        psnr_before = 4.343 * tf.log(9.375 * 9.375 / mse_before)
        psnr = 4.343 * tf.log(9.375 * 9.375 / mse)
        psnr_delta = psnr - psnr_before

        summary_op = tf.summary.merge(
            [
                tf.summary.scalar("mse", mse),
                tf.summary.scalar("rmse", rmse),
                tf.summary.scalar("input rmse", tf.sqrt(mse_before)),
                tf.summary.scalar("delta_PSNR", psnr_delta),
            ]
        )

        return mse, rmse, summary_op


def run_training(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.dataset_size * FLAGS.train_n / FLAGS.batch_size
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
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(
        grads, global_step=global_step
    )  # increments global_step

    return apply_gradient_op


def random_shuffle(x):
    x = np.transpose(x, [1, 2, 0])
    np.random.shuffle(x)
    x = np.transpose(x, [2, 0, 1])
    return x


def main_train():
    # print("GRAPH")
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(
            0, trainable=False, name="global_step"
        )  # steps counter

        x_batch, y_batch, train_init = get_dataset(FLAGS.dataset_train)
        x_valid_batch, y_valid_batch, valid_init = get_dataset(FLAGS.dataset_valid)

        # input placeholder
        with tf.variable_scope("input") as scope:
            x_place = tf.placeholder(
                tf.float32, [FLAGS.batch_size, FLAGS.n_point, 9], "xyz"
            )
            y_place = tf.placeholder(tf.float32, [FLAGS.batch_size], "dz_gt")
            flag_train = tf.placeholder(tf.bool, shape=(), name="flag_train")

        # forward pass
        y_output = run_forward(x_place, flag_train, method=FLAGS.method)

        # loss
        y_loss, y_acc = run_loss(y_output, y_place)
        _, y_valid, summ_valid = run_valid(y_output, y_place)

        # backpropagation
        train_op = run_training(y_loss, global_step)

        # init session
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            train_init,
            valid_init,
        )

        sess = tf.Session()
        sess.run(init_op)

        # summaries
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARY_OP))
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            x_batch_np, y_batch_np = sess.run([x_batch, y_batch])
            theta = np.random.uniform(0, 2 * math.pi, FLAGS.batch_size)
            # theta = random.random()*2*math.pi
            c = np.cos(theta)
            s = np.sin(theta)
            rot_mat = np.zeros((FLAGS.batch_size, 3, 3))
            rot_mat[:, 0, 0] = c
            rot_mat[:, 1, 1] = c
            rot_mat[:, 0, 1] = -s
            rot_mat[:, 1, 0] = s
            rot_mat[:, 2, 2] = 1

            y_batch_np = y_batch_np[:, 0]
            if FLAGS.use_abs:
                y_batch_np = np.abs(y_batch_np)

            for i in range(FLAGS.batch_size):
                x_batch_np[i, :, :3] = np.dot(x_batch_np[i, :, :3], rot_mat[i, :, :].T)
                x_batch_np[i, :, 3:6] = np.dot(
                    x_batch_np[i, :, 3:6], rot_mat[i, :, :].T
                )
                x_batch_np[i, :, 6:] = np.dot(x_batch_np[i, :, 6:], rot_mat[i, :, :].T)

            _, loss_val, acc_val, summary_str = sess.run(
                [train_op, y_loss, y_acc, summary_op],
                feed_dict={x_place: x_batch_np, y_place: y_batch_np, flag_train: True},
            )
            duration = time.time() - start_time
            assert not np.isnan(loss_val), "Model diverged with loss = NaN"

            if step % 10 == 0:
                summary_writer.add_summary(summary_str, step)
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = "{}: step {:d}, rmse = {:9f}, ({:.1f} examples/sec; {:.3f} sec/batch)"
                print(
                    format_str.format(
                        datetime.now(), step, acc_val, examples_per_sec, sec_per_batch
                    )
                )
                # print(format_str.format(datetime.now(), step, acc_val, 0.0, 0.0))

            if step > 0 and step % 100 == 0:
                x_batch_np, y_batch_np = sess.run([x_valid_batch, y_valid_batch])

                y_batch_np = y_batch_np[:, 0]
                if FLAGS.use_abs:
                    y_batch_np = np.abs(y_batch_np)

                acc_val, summary_str = sess.run(
                    [y_valid, summ_valid],
                    feed_dict={
                        x_place: x_batch_np,
                        y_place: y_batch_np,
                        flag_train: False,
                    },
                )
                summary_writer.add_summary(summary_str, step)
                print("valid loss: {:9f}".format(acc_val))

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
        sess.close()


if __name__ == "__main__":
    main_train()
