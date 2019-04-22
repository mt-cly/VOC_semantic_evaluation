import numpy as np
import cv2 as cv
import os


class Color_unit:
    """
    useBGR: the following operations would use such order : B,G,R. else order by R,G,B
    """

    def __init__(self, useBGR=False):
        self.useBGR = useBGR
        self.uint_map = self.get_uint_map()

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    def get_uint_map(self, N=256):
        """

        :param N:
        :return:return a list 'cmap' for mapping the index to the BGR/RGB value. cmap[cid]=[R/B_value,G_value,B/R_value)
        """

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = 0
            g = 0
            b = 0
            id = i
            for j in range(8):
                str_id = self.uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            if self.useBGR:
                cmap[i, 0] = b
                cmap[i, 1] = g
                cmap[i, 2] = r
            else:
                cmap[i, 0] = r
                cmap[i, 1] = g
                cmap[i, 2] = b
        return cmap

    def uint2color(self, uint_img):
        """

        :param uint_img:
        :return:
        """
        return self.uint_map[uint_img[:, :, 0]]

    def color2uint(self, color_img):
        """input the colored img(RGB or BGR), return it's index-form represent"""
        shape = color_img.shape
        color_img = color_img.reshape(-1, 3).tolist()
        list_unit_map = self.uint_map.tolist()

        return np.array([list_unit_map.index(pixel) for pixel in color_img], dtype=np.uint8).reshape(shape[:-1])

    def color2uint_complex(self, color_img):
        """input the colored img(RGB or BGR), return it's index-form represent"""
        shape = color_img.shape
        unit_img = np.zeros(list(shape[0:2]), dtype=np.uint8).reshape(-1)
        if self.useBGR:
            b, g, r = color_img[:, :, 0].reshape(-1), color_img[:, :, 1].reshape(-1), color_img[:, :, 2].reshape(-1)
        else:
            r, g, b = color_img[:, :, 0].reshape(-1), color_img[:, :, 1].reshape(-1), color_img[:, :, 2].reshape(-1)
        for index in range(8):
            unit_img = unit_img << 3
            unit_img = np.array(
                [unit_img[i] ^ ((b[i] & 1) << 2) ^ ((g[i] & 1) << 1) ^ (r[i] & 1) for i in range(len(unit_img))])
            r = r >> 1
            g = g >> 1
            b = b >> 1
        return np.array(unit_img.reshape(shape[0:2]), dtype=np.uint8)


if __name__ == '__main__':
    IMGDIR = '/data/PASCAL_VOC12/VOC2012trainval/JPEGImages'
    SDIR = '/home/lee/CLY/repo/psa_data/semantic/out_rw'
    TDIR = '/home/lee/CLY/repo/psa_data/semantic/results/VOC2012/Segmentation/comp5_val_cls'
    cu_change = Color_unit(useBGR=True)
    for index, name in enumerate(os.listdir(SDIR)):
        print('{}/{}'.format(index, len(os.listdir(SDIR))))
        img = cv.imread(os.path.join(SDIR, name))
        anno = cu_change.uint2color(img)
        cv.imwrite(os.path.join(TDIR, name), anno)
