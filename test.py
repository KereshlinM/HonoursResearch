import math

import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
#from keras import backend
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model




model = load_model('checkpoints2/unet2.hdf5',)
model2 = load_model('checkpoints/unet2.hdf5',)

import matplotlib.pyplot as plt


seismPath = "data/validation/seis/"
faultPath = "data/validation/fault/"
n1,n2,n3=128,128,128
dk = 2
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
fp = model.predict(np.reshape(seis,(1,n1,n2,n3,1))[:, :, 10],verbose=1)
print(fp)
fp = fp[0,:,:,0]
print(fp)
fig = plt.figure(figsize=(20,20))

k3 = 10
imgplot1 = plt.imshow(seis[:,:,k3],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()
imgplot2 = plt.imshow(fp,cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()

seismPath = "data/validation/seis/"
faultPath = "data/validation/fault/"
n1,n2,n3=128,128,128
dk = 2
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
fp = model2.predict(np.reshape(seis,(1,n1,n2,n3,1))[:, :, 10],verbose=1)
print(fp)
fp = fp[0,:,:,0]
print(fp)
fig = plt.figure(figsize=(20,20))
#time slice


#xline slice
k3 = 10
imgplot1 = plt.imshow(seis[:,:,k3],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()
imgplot2 = plt.imshow(fp,cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()