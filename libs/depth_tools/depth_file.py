from . import core
import numpy as np


class DepthFileReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.n_scene = 0
        self.n_camera = 0
        self.f = None
        self.init()

    def init(self):
        self.f = open(self.filename, "rb")
        self.n_camera = int.from_bytes(self.f.read(4), "little")
        self.n_scene = int.from_bytes(self.f.read(4), "little")

    def next_camera(self, cam_name="cam"):
        intrinsics_raw = np.fromfile(self.f, np.float32, 9)
        if len(intrinsics_raw) == 0:
            raise RuntimeError("EOF")
        intrinsics = intrinsics_raw.reshape(3, 3).T
        extrinsics = np.fromfile(self.f, np.float32, 16).reshape(4, 4).T
        width = int.from_bytes(self.f.read(4), "little")
        height = int.from_bytes(self.f.read(4), "little")
        depth = np.fromfile(self.f, np.float32, height * width).reshape(height, width)

        cam = core.DepthCameraPinhole(cam_name, (height, width), intrinsics, extrinsics, depth, depth > 0.0)
        return cam

    def next_scene(self):
        cam_list = []
        for i in range(self.n_camera):
            cam = self.next_camera("camera_{}".format(i))
            cam_list.append(cam)
        return cam_list

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __del__(self):
        self.close()
