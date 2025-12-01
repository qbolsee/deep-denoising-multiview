import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime

import numpy as np
import sklearn
import tensorflow as tf
from tqdm import tqdm

import utils.tf_util as tf_util
from libs import depth_tools, img_tools


class Denoiser(object):
    PATCH = 64
    BATCH_SIZE1 = 128
    BATCH_SIZE2 = 128
    NEIGHBORS = 256

    BASE_DIR = r"."
    DENOISER1_DEFAULT = r"denoiser_z"
    DENOISER2_DEFAULT = r"denoiser_xyz"

    def __init__(self, path_denoiser1=None, path_denoiser2=None, verbose=False):
        if path_denoiser1 is None:
            self.path_denoiser1 = os.path.join(self.BASE_DIR, self.DENOISER1_DEFAULT)
        else:
            self.path_denoiser1 = path_denoiser1
        if path_denoiser2 is None:
            self.path_denoiser2 = os.path.join(self.BASE_DIR, self.DENOISER2_DEFAULT)
        else:
            self.path_denoiser2 = path_denoiser2
        self.graph_stage1 = None
        self.graph_stage2 = None
        self.sess_stage1 = None
        self.sess_stage2 = None
        self.op_stage1 = None
        self.op_stage2 = None
        self.verbose = verbose

    def initialize(self, op_stage1="output/Sub:0", op_stage2="CNN/output:0"):
        # STAGE 1 graph
        graph1 = tf.Graph()
        sess1 = tf.Session(graph=graph1)
        with sess1.as_default(), graph1.as_default():
            ckpt_name = tf.train.latest_checkpoint(self.path_denoiser1)
            saver = tf.train.import_meta_graph(ckpt_name + ".meta")
            saver.restore(sess1, ckpt_name)
            self.graph_stage1 = graph1
            self.sess_stage1 = sess1
            self.op_stage1 = graph1.get_tensor_by_name(op_stage1)

            n1 = np.sum(
                [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
            )

        # STAGE 2 graph
        graph2 = tf.Graph()
        sess2 = tf.Session(graph=graph2)
        with sess2.as_default(), graph2.as_default():
            ckpt_name = tf.train.latest_checkpoint(self.path_denoiser2)
            saver2 = tf.train.import_meta_graph(ckpt_name + ".meta")
            saver2.restore(sess2, ckpt_name)
            self.graph_stage2 = graph2
            self.sess_stage2 = sess2
            self.op_stage2 = graph2.get_tensor_by_name(op_stage2)

            n2 = np.sum(
                [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
            )

    def denoise_depth(self, depth, mask=None):
        if mask is not None:
            depth[~mask] = 0
        pipeline_stage1 = img_tools.WindowedBatchPipeline(
            (self.PATCH, self.PATCH), self.BATCH_SIZE1
        )
        pipeline_stage1.set_image(depth[:, :, np.newaxis])
        for batch_in in pipeline_stage1:
            batch_out = self.sess_stage1.run(
                self.op_stage1,
                feed_dict={"input/flag_train:0": False, "input/img_in:0": batch_in},
            )
            pipeline_stage1.set_current_batch(batch_out)
        depth_out = pipeline_stage1.reconstruct_image()[:, :, 0]
        depth_out[~mask] = 0
        return depth_out

    def denoise_multiview(
        self,
        cam_list,
        skip_stage1=False,
        skip_stage2=False,
        use_patch_2d=True,
        steps_stage2=10,
        alpha=1.0,
        beta=0.5,
    ):
        dt1 = 0
        dt2 = 0

        # STAGE 1
        cam_list_stage1 = [cam.copy() for cam in cam_list]
        if not skip_stage1:
            if self.verbose:
                print("DENOISE STAGE 1")
                gen = tqdm(cam_list_stage1)
            else:
                gen = cam_list_stage1
            t1 = datetime.datetime.now()
            for cam in gen:
                # denoise with stage 1
                mask = cam.get_frame_mask()
                depth = cam.get_frame_depth()
                depth_in = depth.copy()
                depth_in[~mask] = 0
                if use_patch_2d:
                    pipeline_stage1 = img_tools.WindowedBatchPipeline(
                        (self.PATCH, self.PATCH), self.BATCH_SIZE1
                    )
                    pipeline_stage1.set_image(depth_in[:, :, np.newaxis])
                    for batch_in in pipeline_stage1:
                        batch_out = self.sess_stage1.run(
                            self.op_stage1,
                            feed_dict={
                                "input/flag_train:0": False,
                                "input/img_in:0": batch_in,
                            },
                        )
                        pipeline_stage1.set_current_batch(batch_out)
                    depth_out = pipeline_stage1.reconstruct_image()[:, :, 0]
                else:
                    batch_in = depth_in[np.newaxis, :, :, np.newaxis]
                    depth_out_batch = self.sess_stage1.run(
                        self.op_stage1,
                        feed_dict={
                            "input/flag_train:0": False,
                            "input/img_in:0": batch_in,
                        },
                    )
                    depth_out = depth_out_batch[0, :, :, 0]
                depth_out[~mask] = depth[~mask]
                cam.set_frame_depth(depth_out)
            t2 = datetime.datetime.now()
            dt1 = (t2 - t1).total_seconds()

        # STAGE 2
        cam_list_stage2 = [cam.copy() for cam in cam_list_stage1]
        if not skip_stage2:
            # p_keep = 0.8
            p_keep = 1.0

            pcl_tmp = depth_tools.merge_pcl([cam.get_pcl() for cam in cam_list_stage2])
            n_points = pcl_tmp.n_points

            tree = sklearn.neighbors.NearestNeighbors(
                self.NEIGHBORS - 1, leaf_size=16, algorithm="ball_tree", n_jobs=-1
            )
            tree.fit(pcl_tmp.xyz)
            _, knn = tree.kneighbors()

            ind = np.zeros((self.NEIGHBORS,), np.int64)

            pcl_index = pcl_tmp.get_attribute("index")

            j_keep = []

            for j in range(n_points):
                ind[0] = j
                ind[1:] = knn[j]
                index_knn = pcl_index[ind]
                percent = np.sum(index_knn == pcl_index[j]) / self.NEIGHBORS
                if percent <= p_keep:
                    j_keep.append(j)

            dz_prev = np.zeros((n_points,), dtype=np.float32)

            for c in range(steps_stage2):
                pipeline_stage2 = img_tools.BatchPipeline(self.BATCH_SIZE2, True)
                if self.verbose:
                    print("PREPARE STAGE 2")
                pcl_list_stage2 = [cam.get_pcl() for cam in cam_list_stage2]
                for cam, pcl in zip(cam_list, pcl_list_stage2):
                    pcl.set_attribute("xyz_n", cam.get_frame_xyz(apply_mask=True))

                pcl_merge, patch_xyz, patch_attr, patch_j = extract_patches(
                    pcl_list_stage2, cam_list_stage2, knn, j_keep
                )

                pcl_merge_dz = np.zeros((pcl_merge.n_points,), np.float32)
                pcl_patches_dz = np.zeros((len(patch_j),), np.float32)

                pipeline_stage2.set_patches(
                    np.concatenate(
                        (patch_xyz, patch_attr["xyz_n"], patch_attr["ray"]), axis=-1
                    )
                )
                pcl_merge_dz[:] = 0
                pcl_patches_dz[:] = 0
                k = 0
                t1 = datetime.datetime.now()
                if self.verbose:
                    print("DENOISE STAGE 2")
                    gen = tqdm(pipeline_stage2)
                else:
                    gen = pipeline_stage2
                for batch_in in gen:
                    batch_dz = self.sess_stage2.run(
                        self.op_stage2,
                        feed_dict={
                            "input/flag_train:0": False,
                            "input/xyz:0": batch_in,
                        },
                    )
                    n = pipeline_stage2.current_batch_size
                    pcl_patches_dz[k : k + n] = batch_dz[:n]
                    k += n
                t2 = datetime.datetime.now()
                dt2 += (t2 - t1).total_seconds()

                pcl_merge_dz[patch_j] = pcl_patches_dz

                if c > 0:
                    pcl_merge_dz = dz_prev * (1 - alpha) + pcl_merge_dz * alpha
                dz_prev = pcl_merge_dz

                pcl_merge.set_attribute("dz", pcl_merge_dz)
                pcl_list_stage2 = depth_tools.split_pcl(pcl_merge)

                for cam, pcl in zip(cam_list_stage2, pcl_list_stage2):
                    frame_dist = cam.get_frame_dist()
                    mask = cam.get_frame_mask()
                    dz = pcl.get_attribute("dz")

                    if steps_stage2 > 1:
                        frame_dist[mask] += dz * beta
                    else:
                        frame_dist[mask] += dz
                    cam.set_frame_dist(frame_dist)

        return cam_list_stage2, dt1, dt2

    def remove_flying_backup(
        self, cam_list, denoise=True, add_normal=False, threshold=0.2
    ):
        cam_list_res = [cam.copy() for cam in cam_list]

        if self.verbose:
            print("REMOVE FLYING PIXELS")

        if denoise:
            cam_list, _, _ = self.denoise_multiview(cam_list, skip_stage2=True)

        for k in range(len(cam_list)):
            cam = cam_list[k]
            xyz = cam.get_frame_xyz(local=True)
            mask = cam.get_frame_mask()
            h, w = mask.shape

            normal = np.zeros((h, w, 3), dtype=np.float32)
            mask2 = np.zeros((h, w), dtype=np.bool)
            mask2[1:-1, 1:-1] = mask[1:-1, 1:-1]
            mask2[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1]
            mask2[1:-1, 1:-1] &= mask[1:-1, :-2] & mask[1:-1, 2:]

            i_g, j_g = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

            xyz_l = xyz[i_g[mask2], j_g[mask2], :]
            xyz_l_up = xyz[i_g[mask2] - 1, j_g[mask2], :]
            xyz_l_right = xyz[i_g[mask2], j_g[mask2] + 1, :]

            v1 = xyz_l_right - xyz_l
            v2 = xyz_l_up - xyz_l

            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n, axis=1, keepdims=True)
            n_norm[n_norm == 0.0] = 1.0
            n /= n_norm
            n[n[:, 2] > 0.0, :] *= -1
            normal[i_g[mask2], j_g[mask2], :] = n

            ray = cam.get_frame_ray(local=True)

            mask_flying = np.zeros((h, w), dtype=np.bool)
            mask_flying[mask2] = np.sum(-n * ray[mask2, :], axis=1) > threshold

            if add_normal:
                cam_list_res[k].set_attribute("normal", normal)
            cam_list_res[k].set_frame_mask(mask_flying)
        return cam_list_res

    def remove_flying(self, cam_list, denoise=True, add_normal=False, threshold=0.1):
        cam_list_res = [cam.copy() for cam in cam_list]

        if self.verbose:
            print("REMOVE FLYING PIXELS")

        if denoise:
            cam_list, _, _ = self.denoise_multiview(cam_list, skip_stage2=True)

        k_neigh = 16

        if self.verbose:
            iterator = tqdm(range(len(cam_list)))
        else:
            iterator = range(len(cam_list))

        for i in iterator:
            cam = cam_list[i]
            xyz = cam.get_frame_xyz(local=True, apply_mask=True)
            ray = cam.get_frame_ray(local=True, apply_mask=True)
            mask = cam.get_frame_mask()
            n_points = len(xyz)

            tree = sklearn.neighbors.NearestNeighbors(
                k_neigh - 1, leaf_size=8, algorithm="ball_tree", n_jobs=-1
            )

            uv = cam.project_xyz(xyz, local=True)
            tree.fit(uv)
            knn = tree.kneighbors(return_distance=False)

            knn_ext = np.zeros((n_points, k_neigh), dtype=np.int64)
            knn_ext[:, 0] = np.arange(n_points)
            knn_ext[:, 1:] = knn[:, :]

            val_list = np.zeros((n_points,), dtype=np.float32)

            for j in range(n_points):
                xyz_nn = xyz[knn_ext[j, :], :]
                p0, n = planeFit(xyz_nn.T)
                if n[2] > 0.0:
                    n *= -1
                val_list[j] = np.dot(-n, ray[j, :])

            mask_flying = np.copy(mask)
            mask_flying[mask] = val_list > threshold

            cam_list_res[i].set_frame_mask(mask_flying)
        return cam_list_res


def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.reshape(
        points, (np.shape(points)[0], -1)
    )  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], (
        "There are only {} points in {} dimensions.".format(
            points.shape[1], points.shape[0]
        )
    )
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, np.linalg.svd(M)[0][:, -1]


def preprocess(x, y=None, eps=0.0001):
    point_cloud = x[:, :, :3]
    point_cloud_n = x[:, :, 3:6]
    point_ray = x[:, :, 6:9]
    # point_index = x[:, :, 9:]

    stdev = np.sqrt(np.mean(np.square(point_cloud), axis=(1, 2), keepdims=True))
    stdev[stdev < eps] = eps

    x[:, :, :3] = point_cloud / stdev
    x[:, :, 3:6] = point_cloud_n / stdev

    point_ray = point_ray / stdev
    ray_length = np.linalg.norm(point_ray, axis=2, keepdims=True)
    point_ray = point_ray / ray_length
    x[:, :, 6:9] = point_ray

    if y is None:
        return x, stdev[:, 0, 0]
    else:
        y /= stdev[:, :, 0]
        return x, stdev[:, 0, 0], y


def coord_sys(vec_z, vec_other):
    # result: rotation matrix with the given z axis, and y in the provided plane
    vec_x_new = np.cross(vec_other, vec_z)
    vec_x_new /= np.linalg.norm(vec_x_new)
    vec_y_new = np.cross(vec_z, vec_x_new)

    rot_mat = np.concatenate((vec_x_new, vec_y_new, vec_z), axis=0).T
    return rot_mat


def extract_patches(pcl_list, cam_list, knn, j_keep, verbose=True):
    n_pcl = len(pcl_list)

    # build attributes for point clouds
    for i in range(n_pcl):
        pos = cam_list[i].get_position()
        pcl_list[i].compute_ray(pos)

    pcl_merge = depth_tools.merge_pcl(pcl_list, add_index=True)

    k = knn.shape[1] + 1

    ind = np.zeros((k,), np.int64)

    n_patch = len(j_keep)

    xyz_patches = np.zeros((n_patch, k, 3), np.float32)
    attr_patches = {}
    for name in pcl_merge.attributes:
        attr = pcl_merge.get_attribute(name)
        if len(attr.shape) == 1:
            attr_patches[name] = np.zeros((n_patch, k), attr.dtype)
        else:
            attr_patches[name] = np.zeros((n_patch, k, attr.shape[1]), attr.dtype)

    if verbose:
        iterator = tqdm(range(n_patch))
    else:
        iterator = range(n_patch)
    for c in iterator:
        j = j_keep[c]
        i = pcl_merge.get_attribute("index")[j]

        xyz = pcl_merge.xyz[j : j + 1]
        ray = pcl_merge.get_attribute("ray")[j : j + 1]
        cam_transform = cam_list[i].transform
        ray_y_cam = cam_transform[:3, 1]

        rot_mat = coord_sys(ray, ray_y_cam)
        rot_mat_inv = np.linalg.inv(rot_mat)

        # first point is always the central point
        ind[0] = j
        ind[1:] = knn[j]
        xyz_transform = np.dot((pcl_merge.xyz[ind] - xyz), rot_mat_inv.T)

        xyz_patches[c] = xyz_transform
        for name in pcl_merge.attributes:
            attr = pcl_merge.get_attribute(name)[ind]
            if name == "ray":
                attr = np.dot(attr, rot_mat_inv.T)
            if name.startswith("xyz"):
                attr = np.dot(attr - xyz, rot_mat_inv.T)
            attr_patches[name][c] = attr

    return pcl_merge, xyz_patches, attr_patches, j_keep
