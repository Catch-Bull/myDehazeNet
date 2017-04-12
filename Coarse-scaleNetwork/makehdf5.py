import random
from PIL import Image
import numpy as np
import os
import cv2

import h5py


TRANS_DIR = '/home/burglar/date/transForTrain/'
HDF5_NAME = ['hdf5ForTrain', 'hdf5ForTest']
IMAGE_DIR = ['/home/burglar/date/hazeForTrain/', '/home/burglar/date/hazeForTest/']
HDF5_FILE = ['/home/burglar/myDehazeNet/Coarse-scaleNetwork/', '/home/burglar/myDehazeNet/Coarse-scaleNetwork/']
LIST_FILE = ['/home/burglar/myDehazeNet/Coarse-scaleNetwork/list_train.txt', '/home/burglar/myDehazeNet/Coarse-scaleNetwork/list_test.txt']
maxSize = 600

def saveInFile(HDF5Path, hdf5Name, ListPath, Id, datas, scores):
    with h5py.File(HDF5Path + hdf5Name + str(Id) + '.h5', 'a') as f:
            f.create_dataset('data', data = datas)
            f.create_dataset('score', data = scores)
            f.close()


    with open(ListPath, 'a') as f:
            f.write(os.path.abspath(HDF5_FILE[kk] + hdf5Name + str(Id) + '.h5') + '\n')
            f.close()


if __name__ == '__main__':
    print '\nplease wait...'    
    for kk,image_dir in enumerate(IMAGE_DIR):

        file_list = os.listdir(IMAGE_DIR[kk])

        random.shuffle(file_list)

        datas = np.zeros((min(maxSize, len(file_list)), 3, 240, 320))

        scores = np.zeros((min(maxSize, len(file_list)), 1, 240, 320))

        for ii, _file in enumerate(file_list):
            
            xx = cv2.imread(IMAGE_DIR[kk] + _file, 1)

            datas[ii % maxSize, 0, :, :] = \
                np.array(xx[:, :, 0]).astype(np.float32) / 255.0
            datas[ii % maxSize, 1, :, :] = \
                np.array(xx[:, :, 1]).astype(np.float32) / 255.0
            datas[ii % maxSize, 2, :, :] = \
                np.array(xx[:, :, 2]).astype(np.float32) / 255.0
            scores[ii % maxSize, 0, :, :] = \
                np.array(cv2.imread(TRANS_DIR + _file, 0)).astype(np.float32) / 255.0
            if (ii + 1) % maxSize == 0:
                saveInFile(HDF5_FILE[kk], HDF5_NAME[kk], LIST_FILE[kk], (ii + 1) / maxSize, datas, scores)
            elif (ii + 1) == len(file_list):
                saveInFile(HDF5_FILE[kk], HDF5_NAME[kk], LIST_FILE[kk], 1, datas, scores)
    print '\ndone...'