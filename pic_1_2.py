import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

###########################################################################
## 获得附件1中.bmp文件名
path_name = os.path.join("2013B", "pic2")
files2 = os.listdir(path_name)
files = [file for file in files2 if file[-4:] == '.bmp']

## 一次读取图像，并拼接在一起。
for i, file in enumerate(files):
    path_file = os.path.join(path_name, file)
    # img = cv2.imread(path_file,0) #cv2不支持中文路径和名称，
    # 如果要用cv2,请将文件夹“附件1”修改成英文，并将修改代码中相应的路径
    img = plt.imread(path_file)
    # print(i,file)
    if i == 0:
        print(len(files))
        imgs = img
        imgs_3d = np.zeros((img.shape[0], img.shape[1], len(files)))
        imgs_3d[:, :, i] = img
    else:
        imgs = np.hstack((imgs, img))  # 直接拼接，查看乱序结果
        imgs_3d[:, :, i] = img  # 用3D array每一幅图像
print(imgs_3d)
# cv2.imshow('test', imgs)
# cv2.waitKey(0)# 加上这一条和后面的一条，防止jupyter奔溃
# cv2.destroyAllWindows()#
plt.figure(figsize=(12, 16))
plt.imshow(imgs, cmap='gray')
plt.axis(False)
plt.show()

#######################################################################

# =====================================================================

###########################################################################

# 计算两幅图拼接的距离d(i,j), i,j相邻，i前j后
n = len(files)
d = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x = imgs_3d[:, -1, i]
        y = imgs_3d[:, 0, j]
        z = x - y
        d[i, j] = np.linalg.norm(z)
        if i == j:
            d[i, j] = np.inf  # 碎纸片只能和其他碎纸片相邻

###########################################################################
## 贪婪算法求解问题1
print(d)
# 先找到第一个碎纸片，第一列是空白的
for i in range(n):
    temp = imgs_3d[:, 0, i]
    if sum(temp) == 255 * len(temp):
        first = i
        print(first)
        break

# 求矩阵d每一行的最小值所对应的下标
ind = np.argmin(d, axis=1)
print(ind)

sorted_ind = [first]
while len(sorted_ind) < len(ind):
    sorted_ind.append(ind[sorted_ind[-1]])
print(sorted_ind)

#####################################
## 按照贪婪算法的结果拼接

for i in range(n):
    temp = imgs_3d[:, :, sorted_ind[i]]
    if i == 0:
        imgs_recover = temp
    else:
        imgs_recover = np.hstack((imgs_recover, temp))

plt.figure(figsize=(12, 16))
plt.imshow(imgs_recover, cmap='gray')
plt.axis(False)
plt.show()

#######################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

########################################################################
###########################################################################
## 基于优化模型求解问题1， 工具包cvxpy
