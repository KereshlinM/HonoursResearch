import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, MaxPooling3D, Conv3D, UpSampling3D, GlobalAveragePooling3D, Activation, BatchNormalization, ReLU, Conv3DTranspose

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3D(filters, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv3D(filters, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return self.relu(x)

class Downsample(keras.layers.Layer):
    def __init__(self, filters):
        super(Downsample, self).__init__()
        self.conv = Conv3D(filters, kernel_size=3, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)

class Upsample(keras.layers.Layer):
    def __init__(self, filters):
        super(Upsample, self).__init__()
        self.conv = Conv3DTranspose(filters, kernel_size=3, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)

class EFAM(keras.layers.Layer):
    def __init__(self, filters):
        super(EFAM, self).__init__()
        self.global_pool = GlobalAveragePooling3D(keepdims=True)
        self.conv1 = Conv3D(filters // 4, kernel_size=1)
        self.relu = ReLU()
        self.conv2 = Conv3D(filters, kernel_size=1)
        self.sigmoid = Activation('sigmoid')

    def call(self, x_high, x_low):
        # Upsample the low-resolution feature map to match the high-resolution shape
        x_low_upsampled = UpSampling3D(size=(
            x_high.shape[1] // x_low.shape[1],
            x_high.shape[2] // x_low.shape[2],
            x_high.shape[3] // x_low.shape[3]
        ))(x_low)
        # Concatenate high and low resolution features
        x = concatenate([x_high, x_low_upsampled])
        # Attention mechanism
        s = self.global_pool(x)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        # Apply attention
        return x * s

def EAResUNet(input_size=(128, 128, 128, 1)):
    inputs = Input(input_size)
    # Encoder
    x = Conv3D(16, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x1 = ResidualBlock(16)(x)
    x2 = Downsample(32)(x1)
    x2 = ResidualBlock(32)(x2)
    x3 = Downsample(64)(x2)
    x3 = ResidualBlock(64)(x3)
    x4 = Downsample(128)(x3)
    x4 = ResidualBlock(128)(x4)

    # Bottleneck EFAMs
    x4 = EFAM(128)(x4, x3)
    x3 = EFAM(64)(x3, x2)
    x2 = EFAM(32)(x2, x1)

    # Decoder
    x = Upsample(64)(x4)
    x = ResidualBlock(64)(x)
    x = Upsample(32)(x)
    x = ResidualBlock(32)(x)
    x = Upsample(16)(x)
    x = ResidualBlock(16)(x)

    # Final output
    x = Conv3D(1, kernel_size=1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()
    return model
