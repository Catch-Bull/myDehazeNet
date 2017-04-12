import h5py
import numpy as np
import cv2


f = h5py.File('hdf5_test.h5', 'r')
f.keys()
a = f['data'][:]
b = f['score'][:]
f.close()

print a.shape
print b.shape

img1 = cv2.imread('/home/burglar/date/nyutrans_cut/1.png', 1)
img1[:, :, 0] = a[0, 0, :, :]
img1[:, :, 1] = a[0, 1, :, :]
img1[:, :, 2] = a[0, 2, :, :]

img2 = b[0, 0, :, :]

print img2
print img1