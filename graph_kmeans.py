#coding = utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

# 输入分割的长度
part_num = int(input("input the number of grapgh_cut(please more than min parts):"))


def rgb2gray(rgb_img):
    h = rgb_img.shape[0]
    w = rgb_img.shape[1]
    gray = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            gray[i,j] = int(rgb_img[i, j, 0] * 0.299 + rgb_img[i, j, 1] * 0.587 + rgb_img[i, j, 2] * 0.114)
    return gray

def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img

# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img

def Get_seed(dist_transform):
    h = dist_transform.shape[0]
    w = dist_transform.shape[1]
    # 总像素
    landmarkimg = np.zeros((h, w), np.uint8)
    trytime = 1

    # 遍历图片进行二值化处理
    final_threshold = dist_transform.max()
    while(1):
        # 遍历进行连通区域赋值
        landmarkimg[dist_transform >= final_threshold] = 255
        landmarkimg = Two_Pass(landmarkimg, NEIGHBOR_HOODS_4)
        if(trytime == 1):
            trytime += 1
        print("number of grapgh_cut:", landmarkimg.max())
        if (landmarkimg.max() >= part_num) or (final_threshold == 1):
            break
        final_threshold -= 1

    print("final number of grapgh_cut:", landmarkimg.max())
    return np.uint8(landmarkimg)


def Get_thresh(gray, mode=0):
    h = gray.shape[0]
    w = gray.shape[1]
    # 总像素
    m = h*w

    if mode == 0:
        otsuimg = np.zeros((h, w), np.uint8)
        # 初始化各灰度级个数统计参数
        histogram = np.zeros(256, np.int32)
        # 初始化各灰度级占图像中的分布的统计参数
        probability = np.zeros(256, np.float32)
        initial_threshold = 0
        final_threshold = 0
        ### 各个灰度级的个数统计
        for i in range(h):
            for j in range(w):
                s = gray[i, j]
                histogram[s] = histogram[s] + 1
        ### 各灰度级占图像中的分布的统计参数
        for i in range(256):
            probability[i] = histogram[i]/m

        # 计算最佳二值化阈值final_threshold = i
        for i in range(255):
            w0 = w1 = 0  #前景+背景灰度概率
            fgs = bgs = 0    # 前景像素点灰度级总和背景像素点灰度级总和
            # j为当前遍历像素值
            for j in range(256):
                if j <= i:  # 当前i为分割阈值
                    w0 += probability[j]  # 前景像素点占整幅图像的比例累加
                    fgs += j * probability[j]
                else:
                    w1 += probability[j]  # 背景像素点占整幅图像的比例累加
                    bgs += j * probability[j]
            u0 = fgs // w0  # 前景像素点的平均灰度
            u1 = bgs // w1  # 背景像素点的平均灰度
            G = w0*w1*(u0-u1)**2
            if G >= initial_threshold:
                initial_threshold = G
                final_threshold = i
        print("二值化阈值:", final_threshold)
        # 遍历图片进行二值化处理
        if len(np.argwhere(gray>final_threshold)) > len(np.argwhere(gray<final_threshold)):
            for i in range(h):
                for j in range(w):
                    if gray[i, j] > final_threshold:
                        otsuimg[i, j] = 0
                    else:
                        otsuimg[i, j] = 255
        else:
            for i in range(h):
                for j in range(w):
                    if gray[i, j] > final_threshold:
                        otsuimg[i, j] = 255
                    else:
                        otsuimg[i, j] = 0
        return otsuimg

# 腐蚀算法
def img_erode(bin_im, kernel, center_coo=(0, 0)):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    if kernel[center_coo[0], center_coo[1]] == 0:
        raise ValueError("指定原点不在结构元素内！")
    erode_img = np.zeros(shape=bin_im.shape)
    for i in range(center_coo[0], bin_im.shape[0]-kernel_w+center_coo[0]+1):
        for j in range(center_coo[1], bin_im.shape[1]-kernel_h+center_coo[1]+1):
            a = bin_im[i-center_coo[0]:i-center_coo[0]+kernel_w,
                j-center_coo[1]:j-center_coo[1]+kernel_h]  # 找到每次迭代中对应的目标图像小矩阵
            erode_img[i, j] = np.min(a * kernel)  # 若“有重合”，则点乘后最大值为0
    return np.uint8(erode_img)

# 膨胀算法
def img_dilate(bin_im, kernel, center_coo=(0, 0)):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    if kernel[center_coo[0], center_coo[1]] == 0:
        raise ValueError("指定原点不在结构元素内！")
    dilate_img = np.zeros(shape=bin_im.shape)
    for i in range(center_coo[0], bin_im.shape[0] - kernel_w + center_coo[0] + 1):
        for j in range(center_coo[1], bin_im.shape[1] - kernel_h + center_coo[1] + 1):
            a = bin_im[i - center_coo[0]:i - center_coo[0] + kernel_w,
                j - center_coo[1]:j - center_coo[1] + kernel_h]
            dilate_img[i, j] = np.max(a * kernel)  # 若“有重合”，则点乘后最大值为0
    return np.uint8(dilate_img)

# 开运算(先腐蚀后膨胀)
def img_open(bin_im, erope_k, dilate_k, erope_c_c=(0, 0), dilate_c_c=(0, 0)):
    open_img = img_erode(bin_im, erope_k, erope_c_c)
    print("open_img_erode:\n", open_img)
    open_img = img_dilate(open_img, dilate_k, dilate_c_c)
    print("open_img_dilate:\n", open_img)
    return open_img

# 闭运算(先膨胀后腐蚀)
def img_close(bin_im, erope_k, dilate_k, erope_c_c=(0, 0), dilate_c_c=(0, 0)):
    close_img = img_dilate(bin_im, dilate_k, dilate_c_c)
    close_img = img_erode(close_img, erope_k, erope_c_c)
    return close_img

def my_distanceTransform(opening):
    h = opening.shape[0]
    w = opening.shape[1]
    max_rd = np.max((h,w))
    dist_pic = np.uint8(np.zeros(shape=opening.shape))
    for i in range(h):
        for j in range(w):
            if(opening[i][j] != 0):
                rd = 1
                while( dist_pic[i][j] == 0 ):
                    up = i-rd if (i-rd >= 0) else 0
                    down = i+rd if (i+rd < h) else h-1
                    left = j-rd if (j-rd >= 0) else 0
                    right = j+rd if(j+rd < w) else w-1
                    # 上下遍历
                    if(up!=0 and (np.min(opening[up, left:right]) == 0)):
                        dist_pic[i][j] = rd
                    elif(down!= h-1 and (np.min(opening[down, left:right]) == 0)):
                        dist_pic[i][j] = rd
                    elif(left!=0 and (np.min(opening[up:down, left]) == 0)):
                        dist_pic[i][j] = rd
                    elif(right!= w-1 and (np.min(opening[up:down, right]) == 0)):
                        dist_pic[i][j] = rd
                    rd = rd + 1
            print(f"dist_pic[{i}][{j}] = {dist_pic[i][j]}")
    print("距离变换完成\n")
    return dist_pic

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist=[]
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2     #平方
        squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance)
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids
    return changed, newCentroids

def Unknow_mark(markers):
    h = markers.shape[0]
    w = markers.shape[1]
    unknow_markers = np.zeros((h, w), np.uint8)
    print("未知区域标记前:\n", markers)
    print("unknow区域分配中...")
    dataSet = np.argwhere(markers != 1)
    # 初始化中心
    centroids = []
    k = markers.max()-1
    print("markers.max():", markers.max())
    for i in range(2, 100):
        init_ = np.argwhere(markers == i)
        if (len(init_) != 0):
            init_center = np.argwhere(markers == i)[0]
            print(f'mark is {i}: [{init_center[0]}][{init_center[1]}]')
            centroids.append(init_center)
            if(len(centroids) == k):
                break
        else:
            print("delete k")
            k = k - 1
    print("正在区域迭代...")
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k) #调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):   #enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        markers[dataSet[i][0],dataSet[i][1]] = j+2
    return markers


##############################################################################
# binary_img = np.zeros((15, 15, 3), dtype=np.uint8)
# index = [[2, 2], [2, 3], [2, 4], [2, 5],
#          [3, 2], [3, 3], [3, 4], [3, 5],
#         [4, 2], [4, 3], [4, 4], [4, 5],
#         [10, 6], [10, 7], [10, 8], [10, 9],
#          [11, 6], [11, 7], [11, 8], [11, 9],
#          [12, 6], [12, 7], [12, 8], [12, 9],
#          [7, 8]]
# binary_img[binary_img==0] = 255
# for i in index:
#     binary_img[i[0], i[1], :] = np.uint8(0)
#
# src = binary_img
src = cv2.imread('testpic2.jpg')
img = src.copy()
gray = rgb2gray(img)

#灰度二值化
thresh = Get_thresh(gray, mode=0)
# 消除噪声
kernel = np.ones((3, 3), np.uint8)
# 对图片进行开运算
opening = img_open(thresh, kernel, kernel)

# 膨胀
sure_bg = img_dilate(opening, kernel)

# 距离变换
dist_transform = my_distanceTransform(opening)

sure_fg = Get_seed(dist_transform)
print("final number of grapgh_cut:", sure_fg.max())
seed = sure_fg.copy()
sure_fg[sure_fg != 0] = 255
# 获得未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 确保背景是1不是0
markers = seed + 1
print("seed:\n", seed)
# 未知区域标记为0
markers[unknown == 255] = 0
markers3 = Unknow_mark(markers)
markers = markers3

plt.subplot(241), plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)),
plt.title('Original'), plt.axis('off')
plt.subplot(242), plt.imshow(thresh, cmap='gray'),
plt.title('Threshold'), plt.axis('off')
plt.subplot(243), plt.imshow(sure_bg, cmap='gray'),opening
plt.title('Dilate'), plt.axis('off')
plt.subplot(244), plt.imshow(dist_transform, cmap='gray'),
plt.title('Dist Transform'), plt.axis('off')
plt.subplot(245), plt.imshow(sure_fg, cmap='gray'),
plt.title('seed'), plt.axis('off')
plt.subplot(246), plt.imshow(unknown, cmap='gray'),
plt.title('Unknow'), plt.axis('off')
plt.subplot(247), plt.imshow(np.abs(markers), cmap='jet'),
plt.title('Markers'), plt.axis('off')
plt.subplot(248), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
plt.title('Result'), plt.axis('off')

plt.show()
