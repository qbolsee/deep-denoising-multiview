import numpy as np
import sklearn.neighbors
from . import core
from tqdm import tqdm
import random


def coord_sys(vec_z, vec_other):
    # result: rotation matrix with the given z axis, and y in the provided plane
    vec_x_new = np.cross(vec_other, vec_z)
    vec_x_new /= np.linalg.norm(vec_x_new)
    vec_y_new = np.cross(vec_z, vec_x_new)

    rot_mat = np.concatenate((vec_x_new, vec_y_new, vec_z), axis=0).T
    return rot_mat


def closest_points(xyz, ray, j, k):
    d = np.linalg.norm(xyz[j, :] - xyz[:, :], axis=1)
    d_ang = 1+(1-np.inner(ray[j, :3], ray[:, :3]))*(4/2)
    d_tot = d*d_ang

    return np.argpartition(d_tot, k)[:k]


def distance_metric(v1, v2):
    d = np.linalg.norm(v1[:3] - v2[:3])
    d_ang = 1 + (1 - np.inner(v1[3:], v2[3:])) * (4 / 2)
    return d * d_ang


def extract_patches_all(pcl_list, cam_list, k=128, verbose=True, leaf_size=16):
    n_pcl = len(pcl_list)

    # build attributes for point clouds
    for i in range(n_pcl):
        pos = cam_list[i].get_position()
        pcl_list[i].compute_ray(pos)

    pcl_merge = core.merge_pcl(pcl_list, add_index=True)

    tree = sklearn.neighbors.NearestNeighbors(k - 1, leaf_size=leaf_size, algorithm="ball_tree", n_jobs=-1)
    tree.fit(pcl_merge.xyz)
    knn = tree.kneighbors(return_distance=False)

    n_point = len(pcl_merge.xyz)

    mask = np.ones((n_point,), dtype=np.bool)

    i_list = np.arange(n_point)

    n_available = n_point
    ind_patches = []

    ind = np.zeros((k,), np.int64)

    while n_available > 0:
        i_available = i_list[mask]
        i = random.choice(i_available)

        ind[0] = i
        ind[1:] = knn[i]

        ind_patches.append(np.copy(ind))

        mask[ind] = False
        n_available = np.sum(mask)


    n_patch = len(ind_patches)

    xyz_patches = np.zeros((n_patch, k, 3), np.float32)
    attr_patches = {}
    for name in pcl_merge.attributes:
        attr = pcl_merge.get_attribute(name)
        if len(attr.shape) == 1:
            attr_patches[name] = np.zeros((n_patch, k), attr.dtype)
        else:
            attr_patches[name] = np.zeros((n_patch, k, attr.shape[1]), attr.dtype)

    for c, ind in enumerate(ind_patches):
        j = ind[0]
        i = pcl_merge.get_attribute("index")[j]

        xyz = pcl_merge.xyz[j:j + 1]
        ray = pcl_merge.get_attribute("ray")[j:j + 1]
        cam_transform = cam_list[i].transform
        ray_y_cam = cam_transform[:3, 1]

        rot_mat = coord_sys(ray, ray_y_cam)
        rot_mat_inv = np.linalg.inv(rot_mat)

        xyz_transform = np.dot((pcl_merge.xyz[ind] - xyz), rot_mat_inv.T)

        xyz_patches[c] = xyz_transform
        for name in pcl_merge.attributes:
            attr = pcl_merge.get_attribute(name)[ind]
            if name == "ray":
                attr = np.dot(attr, rot_mat_inv.T)
            if name.startswith("xyz"):
                attr = np.dot(attr - xyz, rot_mat_inv.T)
            attr_patches[name][c] = attr

    return xyz_patches, attr_patches, ind_patches


def extract_patches(pcl_list, cam_list, k=128, stride=1, verbose=True, leaf_size=16, use_uv=False):
    n_pcl = len(pcl_list)

    # build attributes for point clouds
    for i in range(n_pcl):
        pos = cam_list[i].get_position()
        pcl_list[i].compute_ray(pos)

    pcl_merge = core.merge_pcl(pcl_list, add_index=True)

    n_points = pcl_merge.n_points
    j_patch = range(0, n_points, stride)
    n_patch = len(j_patch)

    if use_uv:
        projected_uv_list = []
        projected_tree_list = []
        projected_knn_list = []

        for i in range(n_pcl):
            cam = cam_list[i]

            uv = cam.project_xyz(pcl_merge.xyz)
            tree = sklearn.neighbors.NearestNeighbors(k-1, leaf_size=leaf_size, algorithm="ball_tree", n_jobs=-1)
            tree.fit(uv)

            projected_uv_list.append(uv)
            projected_tree_list.append(tree)
            projected_knn_list.append(tree.kneighbors(return_distance=False))
    else:
        tree = sklearn.neighbors.NearestNeighbors(k-1, leaf_size=leaf_size, algorithm="ball_tree", n_jobs=-1)
        tree.fit(pcl_merge.xyz)
        # tree = sklearn.neighbors.NearestNeighbors(k, leaf_size=leaf_size, algorithm="ball_tree", n_jobs=-1, metric=distance_metric)
        # tree.fit(np.concatenate((pcl_merge.xyz, pcl_merge.attr[:, :3]), axis=-1))
        knn = tree.kneighbors(return_distance=False)

    if verbose:
        iterator = tqdm(range(n_patch))
    else:
        iterator = range(n_patch)

    xyz_patches = np.zeros((n_patch, k, 3), np.float32)
    attr_patches = {}
    for name in pcl_merge.attributes:
        attr = pcl_merge.get_attribute(name)
        if len(attr.shape) == 1:
            attr_patches[name] = np.zeros((n_patch, k), attr.dtype)
        else:
            attr_patches[name] = np.zeros((n_patch, k, attr.shape[1]), attr.dtype)

    ind = np.zeros((k,), np.int64)
    for c in iterator:
        j = j_patch[c]
        i = pcl_merge.get_attribute("index")[j]

        xyz = pcl_merge.xyz[j:j+1]
        ray = pcl_merge.get_attribute("ray")[j:j+1]
        cam_transform = cam_list[i].transform
        ray_y_cam = cam_transform[:3, 1]

        rot_mat = coord_sys(ray, ray_y_cam)
        rot_mat_inv = np.linalg.inv(rot_mat)

        # first point is always the central point
        ind[0] = j
        if use_uv:
            ind[1:] = projected_knn_list[i][j]
        else:
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

    return xyz_patches, attr_patches


def write_ply_vec(filename, pcl, length=0.1):
    xyz = pcl.xyz
    ray = pcl.get_attribute("ray")

    f = open(filename, "w")
    f.write("ply\n")
    f.write("format binary_little_endian 1.0\n")
    f.write("element vertex {:d}\n".format(2*pcl.n_points))
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    f.write("element edge {:d}\n".format(pcl.n_points))
    f.write("property int vertex1\n")
    f.write("property int vertex2\n")
    f.write("end_header\n")
    f = open(filename, "ab")

    data = np.zeros(pcl.n_points*3)
    data[0::3] = xyz[:, 0]
    data[1::3] = xyz[:, 1]
    data[2::3] = xyz[:, 2]
    f.write(data[:].astype(np.float32).tobytes('C'))

    data2 = np.zeros(pcl.n_points*3)
    data2[0::3] = xyz[:, 0]+ray[:, 0]*length
    data2[1::3] = xyz[:, 1]+ray[:, 1]*length
    data2[2::3] = xyz[:, 2]+ray[:, 2]*length
    f.write(data2[:].astype(np.float32).tobytes('C'))

    ind_points = np.arange(pcl.n_points, dtype=np.int32)
    data3 = np.zeros(pcl.n_points*2, dtype=np.int32)
    data3[0::2] = ind_points[:]
    data3[1::2] = ind_points[:]+pcl.n_points
    f.write(data3[:].astype(np.int32).tobytes('C'))
    f.close()
