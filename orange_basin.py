import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

segy_file_path = r'real_data/orange_basin.sgy'
st = read(segy_file_path, format='SEGY')
def read_int_from_bytes(bytes_data, start_byte, endian='>'):
    return int.from_bytes(bytes_data[start_byte:start_byte + 4], byteorder='big' if endian == '>' else 'little')

INLINE_BYTE_POSITION = 189
CROSSLINE_BYTE_POSITION = 193

inlines = []
crosslines = []
data = []

for tr in st:
    header = tr.stats.segy.trace_header.unpacked_header
    inline = read_int_from_bytes(header, INLINE_BYTE_POSITION - 1)
    crossline = read_int_from_bytes(header, CROSSLINE_BYTE_POSITION - 1)

    inlines.append(inline)
    crosslines.append(crossline)
    data.append(tr.data)

inlines = np.array(inlines)
crosslines = np.array(crosslines)
data = np.array(data)

unique_inlines = np.unique(inlines)
unique_crosslines = np.unique(crosslines)

n_inlines = len(unique_inlines)
n_crosslines = len(unique_crosslines)
n_samples = data.shape[1]

seismic_cube = np.zeros((n_inlines, n_crosslines, n_samples))

for i, (inline, crossline) in enumerate(zip(inlines, crosslines)):
    inline_idx = np.where(unique_inlines == inline)[0][0]
    crossline_idx = np.where(unique_crosslines == crossline)[0][0]
    seismic_cube[inline_idx, crossline_idx, :] = data[i]

seismic_cube.tofile('real_data/orange_basin.dat')
