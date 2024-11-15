import math

import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
#from keras import backend
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model



model = load_model('check100/unet3.hdf5',)

seismPath = "data/validation/seis/"
faultPath = "data/validation/fault/"
n1,n2,n3=128,128,128
seis = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
fault = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
seis = np.reshape(seis,(n1,n2,n3))
fault = np.reshape(fault,(n1,n2,n3))
seis_mean = np.mean(seis)
seis_std = np.std(seis)
seis = seis-seis_mean
seis = seis/seis_std
seis = np.transpose(seis)
fault = np.transpose(fault)
fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
fp = fp[0,:,:,:,0]
fig = plt.figure(figsize=(20,20))


k3 = 10
imgplot1 = plt.imshow(sies[:,:,k3],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()
imgplot2 = plt.imshow(fp[:,:,k3],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()