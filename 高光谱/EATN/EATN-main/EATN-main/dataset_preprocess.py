import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import argparse
from sklearn.decomposition import PCA
import collections
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser("IP")

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_of_segment', type=int, default=6, help='Divide the data processing process into several stages')
parser.add_argument('--windows', type=int, default=11, help='patche size')
parser.add_argument('--sample', type=int, default=200, help='sample sizes for training')
parser.add_argument('--simple_percent', type=int, default=1, help='sample sizes for training')
parser.add_argument('--pca', type=int, default=0, help='Whether PCA is used')
parser.add_argument('--pca_components', type=int, default=30, help='pca_components')


args = parser.parse_args()

def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:, range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, :, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


#10251
if args.simple_percent == 3:
    sample_200 = [5, 25, 25, 25, 25, 25, 5, 25, 5, 25, 25, 25, 25, 25, 25, 10]  #3%
elif args.simple_percent == 1.5:
    sample_200 = [2, 15, 10, 10, 10, 10, 3, 15, 2, 10, 15, 10, 10, 15, 10, 3]  #1.5%
elif args.simple_percent == 0.5:
    sample_200 = [2, 5, 3, 3, 3, 3, 2, 5, 2, 3, 5, 3, 3, 5, 3, 2]  #0.5%
elif args.simple_percent == 5:
    sample_200 = [2, 72, 42, 12, 24, 36, 2, 24, 2, 48, 120, 30, 20, 62, 18, 4]
elif args.simple_percent == 10:
    sample_200 = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 60, 20, 127, 39, 9]
elif args.simple_percent == 1:
    sample_200 = [4, 10, 6, 6, 6, 6, 4, 10, 4, 6, 10, 6, 6, 10, 6, 4]
elif args.simple_percent == 15:
    sample_200 = [7, 215, 125, 36, 72, 109, 5, 72, 4, 145, 366, 90, 40, 189, 57, 13]

sample_400 = [178, 20, 9, 9, 17, 24, 19, 115, 9]

sample_200 = [2 * i for i in sample_200]

if args.sample == 200:
    SAMPLE = sample_200
elif args.sample == 400:
    SAMPLE = sample_400
else:
    print('Sample size is out of range.')


#分割训练集和测试集
def rSampling(groundTruth, sample_num=SAMPLE):  # divide datasets into train and test datasets
    whole_loc = {}
    train = {}
    val = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        whole_loc[i] = indices
        train[i] = indices[:sample_num[i]]
        val[i] = indices[sample_num[i]:]

    whole_indices = []
    train_indices = []
    val_indices = []
    for i in range(m):
        whole_indices += whole_loc[i]
        train_indices += train[i]
        val_indices += val[i]
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
    return whole_indices, train_indices, val_indices

def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)),
                            'constant', constant_values=0)
    return new_matrix



def split_indices(data,n):
    total_length = len(data)
    each_length = total_length//n

    splited_length = []
    for i in range(n):
        if i !=n-1:
            splited_length.append(each_length)
        elif i == n-1:
            splited_length.append(total_length-each_length*(i))

    return splited_length

def countEachClassInTrain(y_count_train,num_class):
    each_class_num=np.zeros([num_class])
    for i in y_count_train:
        i=int(i)
        each_class_num[i]=each_class_num[i]+1
    return each_class_num

mat_data = sio.loadmat('Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
print(data_IN.shape)
if args.pca == 1:
    data_IN = applyPCA(data_IN, numComponents=args.pca_components)
    print('Data shape after PCA: ', data_IN.shape)
bands = data_IN.shape[-1]
nb_classes = np.max(gt_IN)

# Input datasets configuration to generate 102x9x9 HSI samples
new_gt_IN = gt_IN

# img_rows, img_cols =  7, 7 # 9, 9

INPUT_DIMENSION_CONV = bands
INPUT_DIMENSION = bands


TOTAL_SIZE = np.sum(gt_IN>0)

TRAIN_SIZE = sum(SAMPLE)
print("")

TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

img_channels = bands

PATCH_LENGTH = int(args.windows/2)

MAX = data_IN.max()

data_IN = np.transpose(data_IN, (2, 0, 1))

data_IN = data_IN - np.mean(data_IN, axis=(1, 2), keepdims=True)
data_IN = data_IN / MAX

data = data_IN.reshape(np.prod(data_IN.shape[:1]), np.prod(data_IN.shape[1:]))

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])

padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

CATEGORY = nb_classes

all_indices, train_indices, test_indices = rSampling(gt)

y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1
print('训练集长度(按照label核算)：{}'.format(len(y_train)))
print('测试集长度(按照label核算)：{}'.format(len(y_test)))

print('训练集长度(按照data核算)：{}'.format(len(train_indices)))
print('测试集长度(按照data核算)：{}'.format(len(test_indices)))
splited_train_len = split_indices(train_indices,args.num_of_segment)
print("训练集分段情况：")
print(splited_train_len)
splited_test_len = split_indices(test_indices,args.num_of_segment)
print("测试集分段情况：")
print(splited_test_len)

X_train=np.empty([0,bands,args.windows,args.windows])
X_test=np.empty([0,bands,args.windows,args.windows])

for i in range(args.num_of_segment):
    print("---------分割第{}段---------".format(i + 1))
    X_train_i = np.empty([splited_train_len[i], bands, args.windows, args.windows])
    X_test_i = np.empty([splited_test_len[i], bands, args.windows, args.windows])

    print("start_index:{}".format(i*splited_train_len[1]))
    print("end_index:{}".format(i*splited_train_len[1]+splited_train_len[i]-1))

    train_assign = indexToAssignment(train_indices[i*splited_train_len[1]:i*splited_train_len[1]+splited_train_len[i]-1],
                                     PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])


    for j in range(len(train_assign)):
        X_train_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[j][0], train_assign[j][1])

    test_assign = indexToAssignment(test_indices[i*splited_test_len[1]:i*splited_test_len[1]+splited_test_len[i]-1],
                                    PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for j in range(len(test_assign)):
        X_test_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[j][0], test_assign[j][1])

    X_train=np.vstack((X_train,X_train_i))
    X_test=np.vstack((X_test,X_test_i))

def splitTrainValSet(X, y, testRatio=0.50):
    print("分割前y.size={}".format(y.size))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)
    print("训练集的大小：{}".format(y_train.size))
    print("val集的大小:{}".format(y_val.size))
    return X_train, X_val, y_train, y_val

def savePreprocessedData(X_trainPatches, X_valPatches, X_testPatches, y_trainPatches, y_valPatches, y_testPatches,
                        windowSize):
    if args.pca == 1:
        with open(str(args.simple_percent) + "%XtrainWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open(str(args.simple_percent) + "%XvalWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, X_valPatches)
        with open(str(args.simple_percent) + "%XtestWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open(str(args.simple_percent) + "%ytrainWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open(str(args.simple_percent) + "%yvalWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, y_valPatches)
        with open(str(args.simple_percent) + "%ytestWindowSize" + str(windowSize) + "PCA" + str(args.pca_components) + ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)
    else:
        with open(str(args.simple_percent) + "%XtrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open(str(args.simple_percent) + "%XvalWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, X_valPatches)
        with open(str(args.simple_percent) + "%XtestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open(str(args.simple_percent) + "%ytrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open(str(args.simple_percent) + "%yvalWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, y_valPatches)
        with open(str(args.simple_percent) + "%ytestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)


X_train,X_val,y_train,y_val = splitTrainValSet(X_train,y_train,testRatio=0.5)


print("train集各类别的数量：")
print(countEachClassInTrain(y_train,nb_classes))
print("val集各类别的数量：")
print(countEachClassInTrain(y_val,nb_classes))
print("test集各类别的数量：")
print(countEachClassInTrain(y_test,nb_classes))

savePreprocessedData(X_train, X_val, X_test, y_train, y_val, y_test, windowSize=args.windows)