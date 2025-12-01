from . import core
import h5py
import numpy as np
import math


def read_tof(filename):
    f = h5py.File(filename, "r")
    if "depth" not in f:
        raise ValueError("File not valid")
    h, w = f["depth"].shape
    depth = np.zeros((h, w), np.float32)
    depth[:] = f["depth"][:]
    if "confidence" in f:
        confidence = np.zeros((h, w), np.float32)
        confidence[:] = f["confidence"][:]
    else:
        confidence = None
    hardware = f.attrs["hardware"].decode("ascii")
    focal = f.attrs["focal"]
    center = f.attrs["center"]
    f.close()

    intrinsics = np.array([[focal, 0, center[1]],
                           [0, focal, center[0]],
                           [0, 0, 1]], np.float32)

    extrinsics = np.eye(4, dtype=np.float32)

    if confidence is not None:
        attr = {"confidence": confidence}
    else:
        attr = None

    # mask (conventional)
    mask = depth > 0.0

    camera = core.DepthCamera(hardware, depth, intrinsics, extrinsics, mask, attr)

    return camera


def add_tof_noise(camera, freq=16e6, low_bound=0.1, sigma=0.002):
    d_in = camera.frame_depth
    ir_in = camera.get_attribute("ir")

    # defaults to 9.375m MAX
    lambd = 3e8 / freq
    h, w = np.shape(d_in)

    ir = np.copy(ir_in)
    ir[ir < low_bound] = low_bound
    phase = 2 * math.pi * d_in / (lambd / 2.0)
    z = ir * np.exp(phase * 1j)

    n = sigma * np.random.randn(h, w) + sigma * np.random.randn(h, w) * 1j
    z_noisy = z + n
    phase_noisy = np.angle(z_noisy)
    phase_noisy[phase_noisy < 0.0] += 2 * math.pi

    ir_noisy = np.abs(z_noisy)
    d_noisy = phase_noisy * (lambd / 2.0) / (2 * math.pi)
    d_noisy[~camera.mask] = 0

    camera_noisy = camera.copy()
    camera_noisy.frame_depth = d_noisy
    camera_noisy.set_attribute("ir", ir_noisy[:, :])

    return camera_noisy
