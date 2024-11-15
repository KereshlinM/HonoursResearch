import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
from model_unet3 import *
from model_EAResUnet import *
from model_unet2 import *
import os


class DataGenerator2D_v2(keras.utils.Sequence):
    def __init__(self, data_IDs, batch_size=1, dim=(128, 128, 1), training=True, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_IDs = list(data_IDs)
        self.training = training
        self.slice_indexes = list(range(128))
        self.current_file_index = 0
        self.current_slice_index = 0
        self.on_epoch_end()


    def __len__(self):
        return len(self.data_IDs) * len(self.slice_indexes)

    def __getitem__(self, index):
        if self.current_slice_index >= len(self.slice_indexes):
            self.current_file_index += 1
            self.current_slice_index = 0
            if self.current_file_index >= len(self.data_IDs):
                self.current_file_index = 0

        slice_index = self.slice_indexes[self.current_slice_index]
        self.current_slice_index += 1

        data_ID = self.data_IDs[self.current_file_index]
        X, Y = self.__data_generation(data_ID, slice_index)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_IDs)
        self.current_file_index = 0
        self.current_slice_index = 0
        random.shuffle(self.slice_indexes)

    def __data_generation(self, data_ID, slice_index):
        if self.training:
            seis = np.fromfile(f"data/training/seis/{str(data_ID)}.dat", dtype=np.single)
            fault = np.fromfile(f"data/training/fault/{str(data_ID)}.dat", dtype=np.single)
        else:
            seis = np.fromfile(f"data/validation/seis/{str(data_ID)}.dat", dtype=np.single)
            fault = np.fromfile(f"data/validation/fault/{str(data_ID)}.dat", dtype=np.single)


        # Reshape to 3D (128, 128, 128) and extract 2D slices
        seis = np.reshape(seis, (128, 128, 128))
        fault = np.reshape(fault, (128, 128, 128))

        # Extract the specified slice along the depth axis
        seis_2d = seis[:, :, slice_index]
        fault_2d = fault[:, :, slice_index]

        seis_mean = np.mean(seis_2d)
        seis_std = np.std(seis_2d)
        seis_2d = (seis_2d - seis_mean) / seis_std

        X = np.expand_dims(seis_2d, axis=-1)
        Y = np.expand_dims(fault_2d, axis=-1)
        return np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)


class DataGenerator2D(keras.utils.Sequence):
    def __init__(self, data_IDs, batch_size=1, dim=(512, 512, 1), training=True, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_IDs = data_IDs
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_IDs) / self.batch_size))

    def __getitem__(self, index):
        bsize = self.batch_size
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data_IDs_temp = [self.data_IDs[k] for k in indexes]
        X, Y = self.__data_generation(data_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        if self.training:
            seis = np.fromfile(f"data/training/seis/{str(data_IDs_temp[0])}.dat", dtype=np.single)
            fault = np.fromfile(f"data/training/fault/{str(data_IDs_temp[0])}.dat", dtype=np.single)
        else:
            seis = np.fromfile(f"data/validation/seis/{str(data_IDs_temp[0])}.dat", dtype=np.single)
            fault = np.fromfile(f"data/validation/fault/{str(data_IDs_temp[0])}.dat", dtype=np.single)


        # Reshape to 3D (128, 128, 128) and extract 2D slices
        seis = np.reshape(seis, (128, 128, 128))
        fault = np.reshape(fault, (128, 128, 128))

        # Randomly select a slice along the depth axis
        slice_index = random.randint(0, 127)
        seis_2d = seis[:, :, slice_index]
        fault_2d = fault[:, :, slice_index]

        seis_mean = np.mean(seis_2d)
        seis_std = np.std(seis_2d)
        seis_2d = (seis_2d - seis_mean) / seis_std

        X = np.expand_dims(seis_2d, axis=-1)
        Y = np.expand_dims(fault_2d, axis=-1)
        return np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)


class DataGenerator3D(keras.utils.Sequence):
    def __init__(self, data_IDs, batch_size=1, dim=(128,128,128,1), training = True, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_IDs = data_IDs
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_IDs)/self.batch_size))

    def __getitem__(self, index):
        bsize = self.batch_size
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data_IDs_temp = [self.data_IDs[k] for k in indexes]
        X, Y = self.__data_generation(data_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        if self.training:
            seis = np.fromfile(f"data/training/seis/{str(data_IDs_temp[0])}.dat", dtype=np.single)
            fault = np.fromfile(f"data/training/fault/{str(data_IDs_temp[0])}.dat", dtype=np.single)
        else:
            seis = np.fromfile(f"data/validation/seis/{str(data_IDs_temp[0])}.dat", dtype=np.single)
            fault = np.fromfile(f"data/validation/fault/{str(data_IDs_temp[0])}.dat", dtype=np.single)
        seis = np.reshape(seis, self.dim)
        fault = np.reshape(fault, self.dim)
        seis_mean = np.mean(seis)
        seis_std = np.std(seis)
        seis = np.transpose((seis - seis_mean) / seis_std)
        fault = np.transpose(fault)
        X = np.zeros((2, *self.dim, 1), dtype=np.single)
        Y = np.zeros((2, *self.dim, 1), dtype=np.single)
        X[0,] = np.reshape(seis, (*self.dim, 1))
        Y[0,] = np.reshape(fault, (*self.dim, 1))
        X[1,] = np.reshape(np.flipud(seis), (*self.dim, 1))
        Y[1,] = np.reshape(np.flipud(fault), (*self.dim, 1))
        return X, Y


def trainEAResUnet():
    train_ID = range(10)
    valid_ID = range(4)
    train_generator = DataGenerator(data_IDs=train_ID, dim=(128,128,128,1), training=True, shuffle=True)
    valid_generator = DataGenerator(data_IDs=valid_ID, dim=(128,128,128,1), training=False, shuffle=True)
    model = EAResUnet(input_size=(None,None,None,1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_generator, validation_data=valid_generator, epochs=5, verbose=1)
    model.save('check2/unet3.hdf5')
    print("test")
    showHistory(history)


def trainUnet3():
    train_ID = range(10)
    valid_ID = range(2)
    train_generator = DataGenerator3D(data_IDs=train_ID, dim=(128,128,128,1), training=True, shuffle=True)
    valid_generator = DataGenerator3D(data_IDs=valid_ID, dim=(128,128,128,1), training=False, shuffle=True)
    model = unet3_v2(input_size=(None,None,None,1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_generator, validation_data=valid_generator, epochs=2, verbose=1)
    model.save('checkv1/unet3.hdf5')
    print("test")
    showHistory(history)

def trainUnet2():
    train_ID = range(200)  # Since we only have one data file for now
    valid_ID = range(20)
    train_generator = DataGenerator2D_v2(data_IDs=train_ID, dim=(128, 128, 1), training=True, shuffle=True)
    valid_generator = DataGenerator2D_v2(data_IDs=valid_ID, dim=(128, 128, 1), training=False, shuffle=True)
    model = unet2_v2(input_size=(128, 128, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_generator, validation_data=valid_generator, epochs=5, verbose=1)
    model.save('check2/unet2.hdf5')
    showHistory(history)

def showHistory(history):
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()

trainUnet3()