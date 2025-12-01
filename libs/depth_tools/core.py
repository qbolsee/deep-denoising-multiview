import numpy as np
import copy
import os
import plyfile
import cv2
import json
import glob

import numpy.lib.recfunctions as rfn

import matplotlib.pyplot as plt


class DepthCamera(object):
    def __init__(self, name, yx_res, xy_table=None, extrinsics=None, frame_depth=None, frame_mask=None,
                 attributes=None):
        self.name = name
        self.yx_res = list(yx_res)
        self.xy_table = xy_table
        self.frame_depth = frame_depth
        self.frame_mask = frame_mask
        self.extrinsics = None
        self.transform = None
        self.set_extrinsics(extrinsics)
        if attributes is None:
            self.attributes = {}
        else:
            self.attributes = attributes

    def set_extrinsics(self, extrinsics):
        if extrinsics is None:
            extrinsics = np.eye(4, dtype=np.float32)
        self.extrinsics = extrinsics
        self.transform = np.linalg.inv(extrinsics)
        self.transform /= self.transform[3, 3]

    def copy(self):
        return copy.deepcopy(self)

    def get_frame_xyz(self, local=False, apply_mask=False):
        h, w = self.yx_res
        frame_xyz = np.zeros((h, w, 3), np.float32)
        depth = self.get_frame_depth()
        frame_xyz[:, :, :2] = self.xy_table * depth[:, :, np.newaxis]
        frame_xyz[:, :, 2] = depth[:, :]
        if not local:
            mask_all = np.ones((h, w), np.bool)
            p = frame_xyz[mask_all, :]
            p = _apply_affine(self.transform, p)
            frame_xyz[mask_all, :] = p
        if apply_mask:
            mask = self.get_frame_mask()
            return frame_xyz[mask, :]
        else:
            return frame_xyz

    def get_frame_ray(self, local=False, apply_mask=False):
        h, w = self.yx_res
        frame_ray = np.zeros((h, w, 3), np.float32)
        frame_ray[:, :, :2] = self.xy_table
        frame_ray[:, :, 2] = 1
        norm = np.linalg.norm(frame_ray, axis=2)
        frame_ray /= norm[:, :, np.newaxis]
        if not local:
            mask_all = np.ones((h, w), np.bool)
            p = frame_ray[mask_all, :]
            p = _apply_affine(self.transform, p, infinity=True)
            frame_ray[mask_all, :] = p
        if apply_mask:
            mask = self.get_frame_mask()
            return frame_ray[mask, :]
        else:
            return frame_ray

    def get_attribute(self, name, apply_mask=False):
        if name not in self.attributes:
            raise AttributeError("attribute {} not found".format(name))
        if apply_mask:
            mask = self.get_frame_mask()
            return np.copy(self.attributes[name][mask])
        else:
            return np.copy(self.attributes[name])

    def set_attribute(self, name, value):
        if type(value) is not np.ndarray:
            raise ValueError("Invalid type for attribute, expected numpy ndarray")
        if value.shape[0] != self.yx_res[0] or value.shape[1] != self.yx_res[1]:
            raise ValueError("Attribute shape does not match camera's resolution")
        self.attributes[name] = np.copy(value)

    def get_position(self):
        return self.transform[:3, 3].T

    def get_frame_depth(self):
        mask = self.get_frame_mask()
        return self.frame_depth * mask.astype(np.float32)

    def set_frame_depth(self, frame_depth):
        self.frame_depth = np.copy(frame_depth)

    def get_frame_dist(self):
        frame_xyz = self.get_frame_xyz(local=True)
        frame_dist = np.linalg.norm(frame_xyz, axis=2)
        return frame_dist

    def get_frame_mask(self):
        if self.frame_mask is None:
            return np.ones(self.frame_depth.shape, np.bool)
        else:
            return self.frame_mask

    def set_frame_mask(self, frame_mask):
        self.frame_mask = frame_mask

    def set_frame_dist(self, frame_dist):
        frame_ray = self.get_frame_ray(local=True)
        frame_xyz = frame_ray * frame_dist[:, :, np.newaxis]
        self.set_frame_depth(frame_xyz[:, :, 2])

    def get_pcl(self):
        xyz = self.get_frame_xyz(apply_mask=True)
        pcl_attributes = {}
        for name in self.attributes:
            pcl_attributes[name] = self.get_attribute(name, apply_mask=True)
        pcl = PointCloud(self.name + "_pcl", xyz, pcl_attributes)
        return pcl


class DepthCameraPinhole(DepthCamera):
    def __init__(self, name, yx_res, intrinsics, extrinsics=None,
                 frame_depth=None, frame_mask=None, attributes=None, dist_coeff=None):
        """
        :param name: name of the camera
        :param frame_depth: array [h, w], distance from the camera to the scene (>0)
        :param intrinsics: intrinsics matrix
        :param extrinsics: extrinsics matrix, axis -Z towards the scene, y up
        :param frame_mask: bool array [h, w], valid pixels
        :param attributes: dict of array, each [h, w, c], optional attributes for each pixel
        """
        DepthCamera.__init__(self, name, yx_res, None, extrinsics, frame_depth, frame_mask, attributes)
        self.intrinsics = intrinsics
        if isinstance(dist_coeff, np.ndarray):
            self.dist_coeff = dist_coeff.tolist()
        else:
            self.dist_coeff = dist_coeff
        self.update_xy_table()

    def update_xy_table(self):
        K = self.intrinsics
        height, width = self.yx_res

        # camera-centric coordinates
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        uv = np.stack((u, v), axis=2)
        mask_all = np.ones((height, width), dtype=np.bool)
        if self.dist_coeff is not None:
            uv_list = uv[mask_all, np.newaxis, :]
            dist_coeff = np.array(self.dist_coeff, dtype=np.float32)
            uv2 = cv2.undistortPoints(uv_list, K, dist_coeff, P=K)
            uv[:, :, 0] = uv2[:, 0, 0].reshape(u.shape)
            uv[:, :, 1] = uv2[:, 0, 1].reshape(v.shape)
            # u, v = _apply_dist_coeff(u, v, K, self.dist_coeff)
        p_c = _apply_affine(np.linalg.inv(K), uv[mask_all, :])
        self.xy_table = np.ones((height, width, 2), dtype=np.float32)
        self.xy_table[mask_all, :] = p_c

    def project_xyz(self, xyz, local=False):
        # project a list of 3D points onto the sensor
        K = self.intrinsics
        if local:
            Rt = np.eye(4)
        else:
            Rt = self.extrinsics

        rvec, jacobian = cv2.Rodrigues(Rt[:3, :3])
        tvec = Rt[:3, 3]
        if self.dist_coeff is None:
            dist_coeff = np.zeros((5,), dtype=np.float32)
        else:
            dist_coeff = np.array(self.dist_coeff, dtype=np.float32)
        uv, _ = cv2.projectPoints(xyz, rvec, tvec, K, dist_coeff)
        return uv[:, 0, :]

        # xyz_c = _apply_affine(Rt, xyz)
        # uvw = np.dot(xyz_c, K.T)
        # uv = uvw[:, :2]/uvw[:, 2:]
        # if self.dist_coeff is not None:
        #     u, v = _undo_dist_coeff(uv[:, 0], uv[:, 1], K, self.dist_coeff)
        #     uv = np.stack((u, v), axis=1)
        #
        # return uv

    def save(self, path):
        if path.endswith(".json"):
            dir_name = os.path.dirname(path)
            config_name = path
        else:
            dir_name = path
            config_name = os.path.join(dir_name, "config_" + self.name + ".json")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        conf = {}
        attr = {}
        conf["name"] = self.name
        conf["intrinsics"] = self.intrinsics.tolist()
        conf["extrinsics"] = self.extrinsics.tolist()
        conf["yx_res"] = self.yx_res
        if self.dist_coeff is not None:
            conf["dist_coeff"] = self.dist_coeff
        if self.frame_mask is not None:
            mask_uint8 = self.get_frame_mask().astype(np.uint8) * 255
            conf["mask"] = _save_cam_attr(dir_name, "mask_" + self.name, mask_uint8)
        if self.frame_depth is not None:
            conf["depth"] = _save_cam_attr(dir_name, "depth_" + self.name, self.frame_depth)
        for attr_name in self.attributes:
            attr_val = self.attributes[attr_name]
            attr[attr_name] = _save_cam_attr(dir_name, attr_name + "_" + self.name, attr_val)
        if attr != {}:
            conf["attributes"] = attr
        with open(config_name, "w") as f:
            json.dump(conf, f)


class KinectCamera(DepthCamera):
    def __init__(self, name, xy_table, extrinsics=None, frame_depth=None, frame_mask=None, attributes=None):
        yx_res = xy_table.shape
        self.xy_table = xy_table
        DepthCamera.__init__(self, name, yx_res, xy_table, extrinsics, frame_depth, frame_mask, attributes)


def _apply_dist_coeff(u, v, K, dist_coeff):
    k1, k2, p1, p2, k3 = dist_coeff
    u_c, v_c = K[0:2, 2]
    u_n = (u - u_c) / K[0, 0]
    v_n = (v - v_c) / K[1, 1]
    r2 = u_n * u_n + v_n * v_n
    u_n2 = u_n * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * u_n * v_n + p2 * (r2 + 2 * u_n * u_n)
    v_n2 = v_n * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * v_n * u_n + p2 * (r2 + 2 * v_n * v_n)
    u = u_n2 * K[0, 0] + u_c
    v = v_n2 * K[1, 1] + v_c
    return u, v


def _undo_dist_coeff(u, v, K, dist_coeff):
    uv_cv2 = np.zeros((u.size, 1, 2), dtype=np.float32)
    uv_cv2[:, 0, 0] = u.flatten()
    uv_cv2[:, 0, 1] = v.flatten()
    uv_cv2_new = cv2.undistortPoints(uv_cv2, K, np.array(dist_coeff), P=K)
    u_new = uv_cv2_new[:, 0, 0].reshape(u.shape)
    v_new = uv_cv2_new[:, 0, 1].reshape(v.shape)
    return u_new, v_new


def _apply_affine(mat, xyz_list, infinity=False):
    n, _ = xyz_list.shape
    p = xyz_list
    w = np.ones((n, 1), np.float32)
    if infinity:
        w *= 0
    p = np.dot(np.hstack([p, w]), mat.T)
    if not infinity:
        p /= p[:, -1:]
    return p[:, :-1]


def _save_cam_attr(path, name, val):
    dtype = val.dtype
    shape = val.shape
    if len(shape) == 2:
        channels = 1
    elif len(shape) == 3:
        channels = val.shape[2]
    else:
        raise ValueError("Wrong shape for attribute {}".format(shape))
    if channels == 1 or channels == 3:
        if dtype == np.uint8:
            filename = name + ".png"
        elif dtype == np.float32 or np.float64:
            filename = name + ".exr"
            val = val.astype(np.float32)
        else:
            raise ValueError("Cannot save attribute of type: {}".format(dtype))
        if channels == 3:
            cv2.imwrite(os.path.join(path, filename), val[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(path, filename), val)
    else:
        filename = name + ".npy"
        np.save(os.path.join(path, filename), val)
    return filename


def _load_cam_attr(filename):
    if filename[-3:] in ("png", "exr"):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            return img[:, :, ::-1]
        else:
            return img
    elif filename[-3:] == "npy":
        return np.load(filename)
    else:
        raise ValueError("Unknown file type: {}".format(filename))


def load_depth_scene(dir_path, filter="*.json"):
    cam_list = []
    for filename in glob.glob(os.path.join(dir_path, filter)):
        cam = load_depth_camera(filename)
        cam_list.append(cam)
    return cam_list


def save_depth_scene(cam_list, dir_path):
    for cam in cam_list:
        cam.save(dir_path)


def load_depth_camera(config_path):
    with open(config_path, "r") as f:
        conf = json.load(f)
    dir = os.path.dirname(config_path)
    name = conf["name"]
    yx_res = conf["yx_res"]
    intrinsics = np.array(conf["intrinsics"], dtype=np.float32)
    extrinsincs = np.array(conf["extrinsics"], dtype=np.float32)
    frame_depth = None
    frame_mask = None
    dist_coeff = None
    if "depth" in conf:
        frame_depth = _load_cam_attr(os.path.join(dir, conf["depth"]))
        if len(frame_depth.shape) == 3:
            frame_depth = frame_depth[:, :, 0]
        frame_depth[np.isnan(frame_depth)] = 0
        frame_depth[np.isinf(frame_depth)] = 0
    if "mask" in conf:
        frame_mask = _load_cam_attr(os.path.join(dir, conf["mask"]))
        frame_mask = frame_mask > 0
    if "dist_coeff" in conf:
        dist_coeff = conf["dist_coeff"]
    attributes = {}
    if "attributes" in conf:
        for attr_name, attr_path in conf["attributes"].items():
            attr_val = _load_cam_attr(os.path.join(dir, attr_path))
            attributes[attr_name] = attr_val
    cam = DepthCameraPinhole(name, yx_res, intrinsics, extrinsincs,
                             frame_depth, frame_mask, attributes, dist_coeff)
    return cam


class PointCloud(object):
    def __init__(self, name, xyz=None, attributes=None):
        self.name = name
        self.n_points = 0
        self.n_attr = 0
        self.xyz = None
        if xyz is not None:
            self.set_xyz(xyz)
        if attributes is None:
            self.attributes = {}
        else:
            self.attributes = attributes

    def copy(self):
        return copy.deepcopy(self)

    def write_ply(self, filename, ascii=False, float_color=False):
        n_points, _ = self.xyz.shape

        vert_data = np.zeros((n_points,), dtype=[("x", "float32"),
                                                 ("y", "float32"),
                                                 ("z", "float32")])
        vert_data["x"][:] = self.xyz[:, 0]
        vert_data["y"][:] = self.xyz[:, 1]
        vert_data["z"][:] = self.xyz[:, 2]

        if "normals" in self.attributes:
            normals = self.get_attribute("normals")
            vert_normals = np.zeros((n_points,), dtype=[("nx", "float32"),
                                                    ("ny", "float32"),
                                                    ("nz", "float32")])
            vert_normals["nx"][:] = normals[:, 0]
            vert_normals["ny"][:] = normals[:, 1]
            vert_normals["nz"][:] = normals[:, 2]
            vert_data = rfn.merge_arrays((vert_data, vert_normals), flatten=True)

        if "color" in self.attributes:
            color = self.get_attribute("color")
            if float_color:
                if color.dtype == np.uint8:
                    color = color.astype(np.float32) / 255
                vert_col = np.zeros((n_points,), dtype=[("red", "float32"),
                                                        ("green", "float32"),
                                                        ("blue", "float32")])

            else:
                if color.dtype in (np.float32, np.float64):
                    color = (color * 255).astype(np.uint8)
                vert_col = np.zeros((n_points,), dtype=[("red", "uint8"),
                                                        ("green", "uint8"),
                                                        ("blue", "uint8")])
            vert_col["red"][:] = color[:, 0]
            vert_col["green"][:] = color[:, 1]
            vert_col["blue"][:] = color[:, 2]
            vert_data = rfn.merge_arrays((vert_data, vert_col), flatten=True)

        el = plyfile.PlyElement.describe(vert_data, "vertex")
        dat = plyfile.PlyData([el], text=ascii)
        dat.write(filename)

    def write_xyz(self, filename):
        np.savetxt(filename, self.xyz)

    def load_xyz(self, filename):
        xyz = np.loadtxt(filename)
        self.set_xyz(xyz)

    def compute_ray(self, xyz_view):
        ray = self.xyz - xyz_view
        ray /= np.linalg.norm(ray, axis=1)[:, np.newaxis]

        self.set_attribute("ray", ray)

    def get_attribute(self, name):
        if name not in self.attributes:
            raise AttributeError("attribute {} not found".format(name))
        return self.attributes[name]

    def set_attribute(self, name, value):
        if type(value) is not np.ndarray:
            raise ValueError("Invalid type for attribute, expected numpy ndarray")
        if value.shape[0] != self.xyz.shape[0]:
            raise ValueError("Attribute shape does not match number of points")
        self.attributes[name] = np.copy(value)

    def set_xyz(self, xyz):
        self.xyz = xyz
        self.n_points = xyz.shape[0]


def read_ply(filename):
    dat = plyfile.PlyData.read(filename)
    if dat.elements[0].name != "vertex":
        raise ValueError("Unkown element for point cloud")
    el = dat.elements[0]
    n_points = len(el.data)
    xyz = np.zeros((n_points, 3), dtype=np.float32)
    xyz[:, 0] = el.data["x"]
    xyz[:, 1] = el.data["y"]
    xyz[:, 2] = el.data["z"]
    attr = {}
    if "red" in el.data.dtype.names:
        color_dtype = el.data["red"].dtype
        if color_dtype not in (np.float32, np.uint8):
            raise ValueError("Uknown color format: {}".format(color_dtype))
        color = np.zeros((n_points, 3), dtype=np.uint8)
        if color_dtype == np.float32:
            color[:, 0] = el.data["red"] * 255
            color[:, 1] = el.data["green"] * 255
            color[:, 2] = el.data["blue"] * 255
        else:
            color[:, 0] = el.data["red"]
            color[:, 1] = el.data["green"]
            color[:, 2] = el.data["blue"]
        attr["color"] = color
    for name in el.data.dtype.names:
        if name not in ("x", "y", "z", "red", "green", "blue"):
            attr[name] = el.data[name]
    pcl = PointCloud("point_cloud", xyz, attributes=attr)
    return pcl


def merge_pcl(pcl_list, name="merge_pcl", add_index=True):
    # if len(pcl_list) == 1:
    #     if add_index:
    #         index_merge = np.zeros((pcl_list[0].n_points,), np.int32)
    #         pcl_list[0].set_attribute("index", index_merge)
    #     return pcl_list[0]

    xyz_merge = np.concatenate([pcl.xyz for pcl in pcl_list], axis=0)

    pcl_merge = PointCloud(name, xyz_merge)

    n = len(pcl_list)

    for name in pcl_list[0].attributes:
        try:
            attribute = np.concatenate([pcl.get_attribute(name) for pcl in pcl_list], axis=0)
            pcl_merge.set_attribute(name, attribute)
        except KeyError:
            raise ValueError("Could not find attribute {} in all point cloud".format(name))

    if add_index:
        index_merge = np.concatenate([np.ones((pcl_list[i].n_points,), np.int32) * i for i in range(n)], axis=0)
        pcl_merge.set_attribute("index", index_merge)

    return pcl_merge


def split_pcl(pcl_merge):
    if "index" not in pcl_merge.attributes:
        raise ValueError("index of points not found")
    index = pcl_merge.get_attribute("index")
    n_pcl = np.max(index) + 1
    pcl_list = []
    for i in range(n_pcl):
        ind = index == i
        xyz = pcl_merge.xyz[ind]
        attributes = {}
        for name in pcl_merge.attributes:
            attributes[name] = pcl_merge.get_attribute(name)[ind]
        pcl = PointCloud("pcl_{}".format(i), xyz, attributes)
        pcl_list.append(pcl)
    return pcl_list


def move_along_rays(pcl, camera, delta):
    pos = camera.get_position()
    pcl.compute_ray(pos)

    ray = pcl.get_attribute("ray")

    new_pcl = pcl.copy()
    new_pcl.xyz += ray * delta[:, np.newaxis]

    return new_pcl


def main():
    # test()
    # a = np.array([[1, 2, 3],
    #               [0, 1, 0]], dtype=np.float32)
    # b = np.array([[255, 0, 255],
    #               [0, 255, 0]], dtype=np.uint8)
    # pc = PointCloud("test", a)
    #
    # pc.set_attribute("color", b)
    # pc.write_ply("test.ply")
    # filename = r"D:\Documents\research\R4_multiview_denoising\xyz_denoiser\output\antinous_gt.ply"
    # pcl = read_ply(filename)
    # pcl.write_ply("test2.ply")
    h, w = 32, 32
    depth = np.ones((h, w))
    K = np.array([[64, 0, 16],
                  [0, 64, 16],
                  [0, 0, 1]], dtype=np.float32)
    # dist_coeff = np.array([0.4, 0, 0, 0, 0])
    dist_coeff = np.array([-0.4, 0, 0, 0, 0])
    Rt = np.array([[1, 0, 0, 10],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)  # dist_coeff = None
    cam = DepthCameraPinhole("test", (h, w), K, Rt, frame_depth=depth, dist_coeff=dist_coeff)
    xyz = cam.get_frame_xyz(apply_mask=True)

    # rvec, jacobian = cv2.Rodrigues(cam.extrinsics)
    # tvec = cam.extrinsics[:3, 3]

    print(xyz)

    uv = cam.project_xyz(xyz)
    # uv2, _ = cv2.projectPoints(xyz, rvec, tvec, cam.intrinsics, cam.dist_coeff)
    # uv2 = uv2[:, 0, :]

    # plt.figure()
    # plt.plot(xyz[:, 0], xyz[:, 1], 'b.')

    # print(np.mean(np.linalg.norm(uv2 - uv, axis=1)))

    plt.figure()
    plt.plot(xyz[:, 0], xyz[:, 1], 'b.')

    plt.figure()
    plt.plot(uv[:, 0], uv[:, 1], 'b.')
    # plt.plot(uv2[:, 0], uv2[:, 1], 'r.')

    plt.show()


def main2():
    base_dir = r"D:\Documents\research\R4_multiview_denoising\dataset\blender\output\man_cycles"
    cam_list = load_depth_scene(base_dir)
    pcl_list = [cam.get_pcl() for cam in cam_list]
    pcl_list = pcl_list[0:1]
    pcl_m = merge_pcl(pcl_list)
    pcl_m.write_ply(r"D:\Documents\research\R4_multiview_denoising\dataset\blender\output\man_cycles\cloud.ply")


if __name__ == "__main__":
    main2()
