import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
from EATN import EATN
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')




def loadData():
    # 读入数据
    data = sio.loadmat('../data/GM13.mat')['img']  #读入高光谱数据
    labels = sio.loadmat('../data/GM13.mat')['map']  #   读入标签数据

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))  #   将数据展平为(H*W,C)
    pca = PCA(n_components=numComponents, whiten=True)  #   保留numComponents个主成分
    newX = pca.fit_transform(newX)  #   降维
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))    #   重塑为(H,W,numComponents)

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))#扩充尺寸
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X#中心填充原始数据

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = False):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    # 遍历每个像素,提取patch
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]#中心像素的标签作为patch标签
            patchIndex = patchIndex + 1
    #   移除背景标签
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 16

def create_data_loader():
    # class_num = 2
    # 读入数据
    X, y = loadData()#X是数据而y是标签
    # 用于测试样本的比例,90%用于测试10%用于训练
    test_ratio = 0.99
    # 每个像素周围提取 patch 的尺寸
    patch_size = 5
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 15

    print('Hyperspectral data shape: ', X.shape)#输出数据集大小
    print('Label shape: ', y.shape)#输出类别个数

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)#输出主成分分析之后的数据形状

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    #对高光谱数据依照像素采样生成小立方体,y_all是每个像素的真实标签的集合
    print('Data cube X shape: ', X_pca.shape)#输出像素采样后数据形状
    print('Data cube y shape: ', y.shape)#输出分类的类别个数

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)#划分训练集和测试集
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求(通道优先)
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose其中C=1（单通道），D=pca_components（光谱维度）
    X = X.transpose(0, 4, 3, 1, 2)  # 维度重排：(B, H, W, D, C) → (B, C, D, H, W)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)#全量数据集
    trainset = TrainDS(Xtrain, ytrain)#训练集
    testset = TestDS(Xtest, ytest)#测试集
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=2,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=2,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=2,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = EATN().to(device)
    # 交叉熵损失函数(任务的loss函数)
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()#训练模式,启用dropout等
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)#数据移至设备
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            data = data.squeeze(1)
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #打印每轮损失
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device


def test(device, net, test_loader, return_proba=False):
    """
    一次前向传播，同时支持返回预测类别、真实标签和概率
    return_proba=True时，返回(y_pred_test, y_test, y_proba)
    return_proba=False时，返回(y_pred_test, y_test)
    """
    net.eval()
    y_pred_test = []
    y_test = []
    y_proba = [] if return_proba else None

    with torch.no_grad():  # 关闭梯度计算，节省内存和时间
        for inputs, labels in test_loader:
            inputs = inputs.squeeze(1)
            inputs = inputs.to(device)
            outputs = net(inputs)  # 一次前向传播

            # 提取预测类别（转CPU->numpy）
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            y_pred_test.extend(preds)
            y_test.extend(labels.numpy())  # 真实标签

            # 若需要概率，计算softmax并存储
            if return_proba:
                proba = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                y_proba.extend(proba)

    # 拼接所有批次结果
    y_pred_test = np.array(y_pred_test)
    y_test = np.array(y_test)

    if return_proba:
        y_proba = np.array(y_proba)
        return y_pred_test, y_test, y_proba
    else:
        return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):#计算每类的准确率和平均准确率

    list_diag = np.diag(confusion_matrix)#混淆矩阵对角线:每类正确预测数
    list_raw_sum = np.sum(confusion_matrix, axis=1)#每类总样本数
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))#每类准确率
    average_acc = np.mean(each_acc)#平均准确率
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):#生成完整评估报告

    target_names = ['background','oil']#背景(海水)和目标(溢油)
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)    #   分类报告
    oa = accuracy_score(y_test, y_pred_test)#总体准确率(OA)
    confusion = confusion_matrix(y_test, y_pred_test)#混淆矩阵
    each_acc, aa = AA_andEachClassAccuracy(confusion)#每类准确率
    kappa = cohen_kappa_score(y_test, y_pred_test)#kappa系数

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100


if __name__ == '__main__':

    # 数据准备（不变）
    train_loader, test_loader, all_data_loader, y_all = create_data_loader()

    # 训练模型（不变）
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=100)
    torch.save(net.state_dict(), 'cls_params/EATN_params.pth')
    toc1 = time.perf_counter()

    # 优化核心：一次测试获取所有需要的结果
    tic2 = time.perf_counter()
    # 调用return_proba=True，一次得到预测类别、真实标签和概率
    y_pred_test, y_test, y_proba = test(device, net, test_loader, return_proba=True)
    toc2 = time.perf_counter()

    recall = recall_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_pred_test)

    print(f'recall:{recall*100}, auc:{auc*100}')

    # 后续指标计算（复用一次测试的结果）
    # 1. 计算混淆矩阵和召回率
    confusion = confusion_matrix(y_test, y_pred_test)
    # 2. 计算AUC（直接使用已获取的y_proba，无需重新计算）
    num_classes = len(np.unique(y_test))  # 自动获取类别数（更灵活）
    each_auc = np.zeros(num_classes)
    for cls in range(num_classes):
        y_true_binary = (y_test == cls).astype(int)
        y_proba_binary = y_proba[:, cls]
        if len(np.unique(y_true_binary)) < 2:
            each_auc[cls] = 0.0
        else:
            each_auc[cls] = roc_auc_score(y_true_binary, y_proba_binary) * 100

    # 3. 原有评价指标（OA、AA、Kappa等）
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)

    # 时间统计（不变）
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2

    #写入报告（新增召回率和AUC）
    file_name = "cls_result/classification_report1.txt"
    with open(file_name, 'w') as x_file:
        # 原有内容（不变）
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} recall (%)'.format(recall))
        x_file.write('\n')
        x_file.write('{} AUC (%)'.format(auc))
        x_file.write('\n')
        # 新增：每类准确率、召回率、AUC



        # 原有内容（不变）
        x_file.write('\n{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    #原有可视化（不变）
    get_cls_map.get_cls_map(net, device, all_data_loader, y_all)
