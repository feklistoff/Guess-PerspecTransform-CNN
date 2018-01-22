import cv2
import random
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def warp(pg, pts, p_w=400, p_h=600):
    d_pts = np.array(((0, 0), (p_w, 0), (p_w, p_h), (0, p_h)), np.float32)
    s_pts = np.reshape(pts, (4, 2))
    M = cv2.getPerspectiveTransform(s_pts, d_pts)
    return cv2.warpPerspective(pg, M, (p_w, p_h))


class DataSet:
    def __init__(self, pts, samples, phase='train', batch_size=16, size=128):
        self.samples_per_epoch = (len(samples) // batch_size) * batch_size
        self.sample_paths = samples
        self.phase = phase
        self.batch_size = batch_size
        self.size = size
        self.scaler = MinMaxScaler()
        self.scaled_pts = self.scaler.fit_transform(pts)

    def resize(self, img):
        return cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    def get_next_batch(self):
        if self.phase == 'train':
            while True:
                shuffle(self.sample_paths, self.scaled_pts)
                for offset in range(0, self.samples_per_epoch, self.batch_size):
                    batch_sample_paths = self.sample_paths[offset:offset+self.batch_size]
                    batch_pts = self.scaled_pts[offset:offset+self.batch_size]
                    batch_imgs = []
                    for i in range(self.batch_size):
                        img = mpimg.imread(batch_sample_paths[i])
                        img = self.resize(img)
                        batch_imgs.append(img)
                    batch_imgs = np.asarray(batch_imgs)
                    batch_pts = np.asarray(batch_pts)
                    yield shuffle(batch_imgs, batch_pts)
        if self.phase == 'test':
            while True:
                for offset in range(0, self.samples_per_epoch, self.batch_size):
                    batch_sample_paths = self.sample_paths[offset:offset+self.batch_size]
                    batch_pts = self.scaled_pts[offset:offset+self.batch_size]
                    batch_imgs = []
                    for i in range(self.batch_size):
                        img = mpimg.imread(batch_sample_paths[i])
                        batch_imgs.append(img)
                    batch_imgs = np.asarray(batch_imgs)
                    batch_pts = np.asarray(batch_pts)
                    yield (batch_imgs, batch_pts)
