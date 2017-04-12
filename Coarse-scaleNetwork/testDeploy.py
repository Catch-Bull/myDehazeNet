import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
caffe_root = '/home/burglar/caffe'
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
import math


caffe.set_mode_cpu()

MEAN_VALUE = 0
IMAGE_SIZE = (100, 100)
net_file = '/home/burglar/myDehazeNet/Coarse-scaleNetwork/deploy.prototxt'
caffe_model = '/home/burglar/myDehazeNet/Coarse-scaleNetwork/example_ising_iter_100.caffemodel'

net = caffe.Net(net_file, caffe_model, caffe.TEST)
net.blobs['data'].reshape(1, 3, *IMAGE_SIZE)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([MEAN_VALUE, MEAN_VALUE, MEAN_VALUE]))
transformer.set_raw_scale('data', 255)

"""
filename = '/home/burglar/date/nyuhaze_cut/690.png'
filename_trans = '/home/burglar/date/nyutrans_cut/690.png'
"""
filename = '/home/burglar/date/hazeForTrain/1.png'
filename_trans = '/home/burglar/date/transForTrain/1.png'
image = caffe.io.load_image(filename, True)
transformed_image = transformer.preprocess('data', image)

image = transformed_image

print image.shape

for i in range(0, 100):
    for j in range(0, 100):
        image[0, i, j] /= 255.0
        image[1, i, j] /= 255.0
        image[2, i, j] /= 255.0

net.blobs['data'].data[...] = image

print image

output = net.forward()
pool = net.blobs['pred'].data[0: 1]
print pool.shape


out = pool[0, 0, :, :] * 255.0
print out

print "---------------------------------------------------"

#pool1 = net.blobs['pool1'].data[0: 1]
#print pool1[0, 0, :, :]


testimg = np.array(cv2.imread(filename_trans, 0)).astype(np.float32)
print testimg


cv2.imshow("my_t", out)
cv2.imshow("real_t", testimg)
cv2.waitKey(0)
cv2.imwrite("my_t.png", out)
cv2.imwrite("real_t.png", testimg)

"""
res = 0.0
for i in range(0, 100):
    for j in range(0, 100):
        res += ((pool[0, 0, i, j] - testimg[i, j]) * (pool[0, 0, i, j] - testimg[i, j]));

res /= (100 * 100)
res = math.sqrt(res)


print res
"""
"""
img = pool[0, 0, :, :] * 255
cv2.imshow("1", img)
cv2.waitKey(0)
#upsam = net.blobs['upsample1'].data[0: 1]
#print upsam[0, 0, :, :]
#print "-----------------------------------------------------"

#upsams = net.params['conv4'][0].data
#print upsams
"""