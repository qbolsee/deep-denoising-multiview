import numpy as np
import math
from enum import Enum


class BorderType1D(Enum):
    START = 0
    MIDDLE = 1
    END = 2


def generate_window_1d(N, border_type):
    x = np.arange(N)[:, np.newaxis]
    x_c = (N-1)/2

    win = 0.5*(1+np.cos(2*math.pi * (x-x_c)/N))

    if border_type == BorderType1D.START:
        win[:N//2] = 1.0
    elif border_type == BorderType1D.END:
        win[N//2:] = 1.0
    return win


class BatchPipeline(object):
    def __init__(self, batch_size, batch_size_fixed=False):
        self.batch_size = batch_size
        self.batch_size_fixed = batch_size_fixed
        self.patches = None
        self.k = 0
        self.current_batch_size = 0

    def set_patches(self, patches):
        self.patches = np.copy(patches)

    def get_patches(self):
        return self.patches

    def iterate_batches(self):
        n_tot = self.patches.shape[0]
        n = self.batch_size

        self.k = 0
        while self.k < n_tot:
            slice_in = self.patches[self.k:self.k + n]
            n_slice = slice_in.shape[0]
            self.current_batch_size = n_slice
            if self.batch_size_fixed and n_slice < n:
                pad_length = n - n_slice
                shape_example = slice_in.shape[1:]
                pad_data = np.zeros((pad_length,)+shape_example, dtype=slice_in.dtype)
                slice_in = np.concatenate((pad_data, slice_in), axis=0)
            yield slice_in
            self.k += n

    def __iter__(self):
        return self.iterate_batches()

    def __len__(self):
        return math.ceil(len(self.patches)/self.batch_size)

    def set_current_batch(self, slice_out):
        n = self.current_batch_size
        self.patches[self.k:self.k+n] = slice_out[:n]


class WindowedBatchPipeline(BatchPipeline):
    def __init__(self, patch_shape, batch_size, batch_size_fixed=False):
        self.patch_shape = patch_shape
        self.windows_2d = {}
        self.patches = None
        self.image = None
        self._generate_windows()

        h_p, w_p = self.patch_shape
        if h_p % 2 != 0 or w_p % 2 != 0:
            raise ValueError("Patch size not divisible by 2")
        BatchPipeline.__init__(self, batch_size, batch_size_fixed)

    def _generate_windows(self):
        h, w = self.patch_shape
        for i in BorderType1D:
            for j in BorderType1D:
                win = np.ones((h, w))
                win *= generate_window_1d(h, i)
                win *= generate_window_1d(w, j).T
                self.windows_2d[(i,j)] = win

    def get_border_type(self, i, i_max, i_step):
        is_start = i == 0
        is_end = i+i_step == i_max
        if is_start and is_end:
            return BorderType1D.MIDDLE
        if is_start:
            return BorderType1D.START
        if is_end:
            return BorderType1D.END
        return BorderType1D.MIDDLE

    def set_image(self, img):
        self.image = np.copy(img)

        img_shape = self.image.shape
        img_dtype = self.image.dtype
        h, w = img_shape[:2]
        h_p, w_p = self.patch_shape
        h_p2, w_p2 = h_p//2, w_p//2
        h_pad = 0 if h % h_p2 == 0 else h_p2 - (h % h_p2)
        w_pad = 0 if w % w_p2 == 0 else w_p2 - (w % w_p2)
        n_patches = ((h+h_pad-h_p2)//h_p2)*((w+w_pad-w_p2)//w_p2)

        no_channel = len(img_shape) == 2
        if no_channel:
            img_padded = np.pad(self.image, ((0, h_pad), (0, w_pad)), "reflect")
            self.patches = np.zeros((n_patches, h_p, w_p), dtype=img_dtype)
            self.patches = np.zeros((n_patches, h_p, w_p), dtype=img_dtype)
        else:
            img_padded = np.pad(self.image, ((0, h_pad), (0, w_pad), (0, 0)), "reflect")
            self.patches = np.zeros((n_patches, h_p, w_p, img_shape[2]), dtype=img_dtype)
            self.patches = np.zeros((n_patches, h_p, w_p, img_shape[2]), dtype=img_dtype)

        k = 0
        for i in range(0, h+h_pad-h_p2, h_p2):
            for j in range(0, w+w_pad-w_p2, w_p2):
                self.patches[k] = img_padded[i:i+h_p, j:j+w_p]
                k += 1

    def reconstruct_image(self):
        img_shape = self.image.shape
        img_dtype = self.image.dtype
        h, w = img_shape[:2]
        h_p, w_p = self.patch_shape
        h_p2, w_p2 = h_p // 2, w_p // 2
        h_pad = 0 if h % h_p2 == 0 else h_p2 - (h % h_p2)
        w_pad = 0 if w % w_p2 == 0 else w_p2 - (w % w_p2)

        no_channel = len(img_shape) == 2

        if no_channel:
            img_padded = np.zeros((h+h_pad, w+w_pad), dtype=img_dtype)
        else:
            img_padded = np.zeros((h+h_pad, w+w_pad, img_shape[2]), dtype=img_dtype)

        k = 0
        i_max = h+h_pad-h_p2
        j_max = w+w_pad-w_p2
        for i in range(0, i_max, h_p2):
            for j in range(0, j_max, w_p2):
                type_i = self.get_border_type(i, i_max, h_p2)
                type_j = self.get_border_type(j, j_max, w_p2)
                win = self.windows_2d[(type_i, type_j)]
                if no_channel:
                    win_new = win
                else:
                    win_new = win[:, :, np.newaxis]
                img_padded[i:i+h_p, j:j+w_p] += self.patches[k] * win_new
                k += 1
        self.image[:, :] = img_padded[:h, :w]

        return self.image

