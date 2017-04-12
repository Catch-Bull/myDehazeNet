
from pylab import *
import matplotlib.pyplot as plt

import math
import caffe

# caffe.set_device(0)
# caffe.set_mode_gpu()

caffe.set_mode_cpu()
solver = caffe.SGDSolver('solver.prototxt')

niter = 100

display_iter = 1

test_iter = 1

test_interval = 1

train_loss = zeros(ceil(niter * 1.0 / display_iter))

test_loss = zeros(ceil(niter * 1.0 / test_interval))

solver.step(1)

_train_loss = 0; _test_loss = 0

for it in range(niter):
    print it
    solver.step(1)
    _train_loss += solver.net.blobs['loss'].data
    if it % display_iter == 0:
        train_loss[it // display_iter] = math.sqrt((_train_loss / display_iter) * 2.0 / 100.0 / 100.0)
        _train_loss = 0

    if it % test_interval == 0:
        for test_it in range(test_iter):
            solver.test_nets[0].forward()
            _test_loss += solver.test_nets[0].blobs['loss'].data
        test_loss[it / test_interval] = math.sqrt((_test_loss / test_iter) * 2.0 / 100.0 / 100.0)
        _test_loss = 0

print '\nplot the train loss and\n'
_, ax1 = plt.subplots()
ax2 = ax1.twinx()


ax1.plot(display_iter * arange(len(train_loss)), train_loss, 'g')

ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')


ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')

#plt.show()

print train_loss