import scipy.io as sio
import numpy as np
labels = sio.loadmat('../data/GM13.mat')['map']
print("Original labels:", np.unique(labels))