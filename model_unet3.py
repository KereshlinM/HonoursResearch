import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, MaxPooling3D, Conv3D, UpSampling3D


def unet3_v1(pretrained_weights = None, input_size = (None,None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv3)

    up4 = concatenate([UpSampling3D(size=(2,2,2))(conv3), conv3], axis=-1)
    conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up4)
    conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv2], axis=-1)
    conv5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv5)

    conv6 = Conv3D(1, (1,1,1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])
    model.summary()
    return model

def unet3_v2(pretrained_weights = None, input_size = (None,None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1,1,1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    model.summary()
    return model

def unet3_v2(pretrained_weights = None, input_size = (None,None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1,1,1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    model.summary()
    return model

def unet3_v3(pretrained_weights = None, input_size = (None,None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(32, (6,6,6), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (6,6,6), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(64, (6,6,6), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (6,6,6), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(128, (6,6,6), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (6,6,6), activation='relu', padding='same')(conv3)

    up4 = concatenate([UpSampling3D(size=(2,2,2))(conv3), conv3], axis=-1)
    conv4 = Conv3D(64, (6,6,6), activation='relu', padding='same')(up4)
    conv4 = Conv3D(64, (6,6,6), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv2], axis=-1)
    conv5 = Conv3D(32, (6,6,6), activation='relu', padding='same')(up5)
    conv5 = Conv3D(32, (6,6,6), activation='relu', padding='same')(conv5)

    conv6 = Conv3D(1, (1,1,1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])
    model.summary()
    return model