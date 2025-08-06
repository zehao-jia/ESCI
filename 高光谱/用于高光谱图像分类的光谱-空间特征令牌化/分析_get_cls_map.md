## get_classification_map(y_pred, y)
```python
def get_classification_map(y_pred, y):

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
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels
```
- **函数作用:将模型预测结果映射到与原图像相同的二维空间**
- *输入:y_pred-所有的非背景像素, y:经过处理的分类图像*
> 1. 创建一个和图像大小一致的全零矩阵,k相当于一个指针指向储存非背景像素存储列表
> 2. 遍历整个图像,将跳过背景像素,将非背景像素传入全零矩阵
> 3. 处理后,返回映射后的图像

## list_to_colormap(x_list)
```python
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):#遍历输入函数的分类结果,根据类别给像素逐个着色
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
    return y
```
- **函数作用:将分类结果映射为颜色展示**
- *输入:x_list:展平为1维数组的分类结果*
>1. 创建与数组等长,宽为3(rgb)的二维数组
>2. 遍历分类结果的每个元素
>3. 若元素类别为0(背景元素),着色为黑色
>4. 若元素类别为1(溢油),着色为红色
>5. 返回二维数组

## classification_map(map, ground_truth, dpi, save_path)
```python
def classification_map(map, ground_truth, dpi, save_path):

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
```
- **函数作用:将分类结果保存为无边框,无坐标轴的图片**
- *输入:map:待可视化的分类结果, ground_truth:原始标签图像, dpi:图片分辨率, save_path:图像保存路径*
> 1. 创建一个无边框图像
> 2. 将图像的长和宽分别乘以二再除以分辨率,确保图像在不同分辨率下依然能还原原始图像的像素比例
> 3. 创建覆盖整个图像的坐标轴
>> - plt.Axes(fig, [left, right, width, height])函数用于设置坐标轴列表中的四个函数分别表示最下交的起始位置和宽高占整个图片比重
> 4.隐藏边框,隐藏坐标轴

## test()
```python
def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0

    for inputs, labels in test_loader:

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
```
- **作用:评估模型性能,搜集预测结果和真实标签**
- *输入变量:运行设备,训练好的模型,测试数据加载器*
>1. 初始化:初始设置count为0,将模型设置为评估模式,y_pred_test和y_test分别用于存储预测的标签和真实标签
```python
 inputs = inputs.to(device)
 outputs = net(inputs)
```
>2. 遍历测试集,对于test_loader中的每个batch,先将其置于gpu训练出结果
```python
outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
```
>3. 获取类别索引
>>- .detach(): 将张量从计算图中分离出来,防止影响梯度计算
>>- .cpu(): 将张量从gpu中移到cpu中
>>- .numpy(): 将张量转换为numpy数据的格式
>>- np.argmax(..., axis=1): 从每行中找出数值最大的元素的列对应的列索引
```python
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

```
>如果count为0(是首个batch),直接赋值,如果不是,利用np.concatenate()拼接
>> np.concatenate():将多个numpy数组组合到一起,没有指定axis则沿着第一个轴(axis=0)拼接
>4. 返回预测标签y_pred_test和真实标签y_test

## get_cls_map(net, device, all_data_loader, y)
```python
def get_cls_map(net, device, all_data_loader, y):

    y_pred, y_new = test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))#将预测结果重塑为原先形状
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))#将真实结果重塑为原先形状

    classification_map(y_re, y, 300,
                       'classification_maps/' + 'IP_predictions3.eps')#预测结果的彩色分类图(矢量图)
    classification_map(y_re, y, 300,
                       'classification_maps/' + 'IP_predictions3.png')#预测结果的彩色分类图(像素图)
    classification_map(gt_re, y, 300,
                       'classification_maps/' + 'IP_gt3.png')#真实标签的彩色分类图
    print('------Get classification maps successful-------')
```
- **函数作用:结果可视化**
- *函数输入:net:已训练好的模型, device:计算设备, all_data_loader:包含全量样本的dataloader, y:原始标签矩阵*
>1. 用test函数得出全量样本的预测标签和真实标签
```python
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()
```
>2. 将预测结果映射为二维矩阵,随后将预测矩阵和真实标签分别展平
>> - get_classification_map(y_pred, y):将预测结果映射为二维图像
>> - np.ravel:将多维数组展平为一维数组(视图)(浅拷贝(不复制,只添加指针))
>> - flatten:将多维数组展平为一维数组(深拷贝)
```python
    y_list = list_to_colormap(x)#将预测的结果向量转化为rgb像素数组
    y_gt = list_to_colormap(gt)#将真实值转化为rgb像素数组
```
>3. 将预测结果和真实值分别转换为rgb像素数组
```python
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
```
>4. 将像素数组重塑为原图像形状
>5. 将预测结果和真实值分别存储