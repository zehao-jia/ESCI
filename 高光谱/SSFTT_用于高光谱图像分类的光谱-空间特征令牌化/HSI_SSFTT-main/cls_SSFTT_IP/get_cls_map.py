import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y):#将模型预测结果映射到与原始图像相同的二维空间,这里的y_pred是展平的向量,y是图像

    height = y.shape[0]#图片的高
    width = y.shape[1]#图片的宽
    k = 0
    cls_labels = np.zeros((height, width))#创建一个二维平面,大小与预测图片相同
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])#目标像素的生成标签,y的值来自模型
            if target == 0:#跳过标签为0的背景像素
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1#+1是因为0被用作背景
                k += 1

    return  cls_labels

def list_to_colormap(x_list):#颜色映射,将分类结果作为颜色展示
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):#遍历输入函数的分类结果,根据类别给像素逐个着色
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    """创建一个无边框无坐标轴的图像,其中,map是待可视化的分类结果,ground_truth是原始标签图像,dpi是分辨率,save_path是保存路径"""
    fig = plt.figure(frameon=False)#创建一个无边框图像
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)#根据原始图像尺寸设置图像大小

    ax = plt.Axes(fig, [0., 0., 1., 1.])#创建覆盖整个图像区域的坐标轴
    ax.set_axis_off()#隐藏坐标轴和边框,让输出只包含分类结果
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)#将分类结果映射为彩色图像
    fig.savefig(save_path, dpi=dpi)#按照指定分辨率将结果保存在指定路径中

    return 0

def test(device, net, test_loader):#测试函数,作用是评估模型性能,搜集预测结果和真实标签
    count = 0
    # 模型测试
    net.eval()#将模型设置为评估模式
    y_pred_test = 0
    y_test = 0

    for inputs, labels in test_loader:#将输入数据转移至指定设备,通过模型预测结果,并使用argmax获取类别索引
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        #逐个拼接预测结果和真实标签
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test#其中,y_pred_test是预测标签,y_test是真实标签

def get_cls_map(net, device, all_data_loader, y):
    """从左到右输入函数的分别是已训练好的模型,计算设备,包含全量样本的dataloader,原始标签矩阵"""

    y_pred, y_new = test(device, net, all_data_loader)#y_pred是对全量样本的预测标签,y_new是真实标签
    cls_labels = get_classification_map(y_pred, y)#将预测结果映射回原始图像二维矩阵
    x = np.ravel(cls_labels)#预测结果的展平向量
    gt = y.flatten()#原始标签的展平向量

    y_list = list_to_colormap(x)#将预测的结果向量转化为rgb像素数组
    y_gt = list_to_colormap(gt)#将真实值转化为rgb像素数组

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))#将预测结果重塑为原先形状
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))#将真实结果重塑为原先形状
    classification_map(y_re, y, 300,
                       'classification_maps/' + 'IP_predictions.eps')#预测结果的彩色分类图(矢量图)
    classification_map(y_re, y, 300,
                       'classification_maps/' + 'IP_predictions.png')#预测结果的彩色分类图(像素图)
    classification_map(gt_re, y, 300,
                       'classification_maps/' + 'IP_gt.png')#真实标签的彩色分类图
    print('------Get classification maps successful-------')