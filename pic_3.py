import os
import numpy as np
import matplotlib.pyplot as plt

###########################################################################
## 获得附件1中.bmp文件名
path_name = os.path.join("2013B", "pic3")
files2 = os.listdir(path_name)
files = [file for file in files2 if file[-4:] == '.bmp']

## 一次读取图像，并拼接在一起。
all_imgs = []

imgs_3d = np.zeros((180, 72, len(files)))
img_3d_01 = np.zeros((180, 72, len(files)))

for i, file in enumerate(files):
    path_file = os.path.join(path_name, file)
    # img = cv2.imread(path_file,0) #cv2不支持中文路径和名称，
    # 如果要用cv2,请将文件夹“附件1”修改成英文，并将修改代码中相应的路径
    img = plt.imread(path_file)
    # print(i,file)
    imgs_3d[:, :, i] = img

print(imgs_3d.shape)

# 图片0-1化
for i in range(imgs_3d.shape[0]):
    for k in range(imgs_3d.shape[1]):
        for j in range(imgs_3d.shape[2]):
            if imgs_3d[i, k, j] >= 170:
                img_3d_01[i, k, j] = 1

print(img_3d_01.shape)

# 每张图片的每一行叠加
line_add_array = np.zeros((180, 209))
for i in range(line_add_array.shape[0]):
    for j in range(line_add_array.shape[1]):
        line_add_array[i, j] = sum(img_3d_01[i, :, j])

print(line_add_array.shape)

# 叠加=72 == 1；叠加<72 == 0
line_add_array_01 = np.zeros((180, 209))
for i in range(line_add_array_01.shape[0]):
    for j in range(line_add_array_01.shape[1]):
        if line_add_array[i, j] == 72:
            line_add_array_01[i, j] = 1
        else:
            line_add_array_01[i, j] = 0

print(line_add_array_01.shape)

# 找出首行
first_line = []
for i in range(209):
    f = 0
    l = 0
    for j in range(5):
        f = f + sum(img_3d_01[:, j, i])
    for j in range(12):
        l = l + sum(img_3d_01[:, 60 + j, i])
    if f == 900 and l != 540:
        first_line.append(i)
print(first_line)

eigenvalue_dict = {i: [] for i in range(len(first_line))}

# 取特征值,用字典保存特征
for i in range(len(first_line)):
    for j in range(line_add_array_01.shape[0] - 1):
        if line_add_array_01[j, first_line[i]] != line_add_array_01[j + 1, first_line[i]]:
            eigenvalue_dict[i].append(j)
print(eigenvalue_dict)
print('===============================')


def min_var(li):
    li2 = []
    for i in range(19):
        index_i = li.index(min(li))  # 得到列表的最小值，并得到该最小值的索引
        li2.append(index_i)  # 记录最小值索引
        li[index_i] = float('inf')  # 将遍历过的列表最小值改为无穷大，下次不再选择
    return li2


all_li = []
for i in range(len(first_line)):
    li = []
    for j in range(209):
        x = eigenvalue_dict[i][0]
        y = eigenvalue_dict[i][-2]
        if i == 7:
            x = eigenvalue_dict[i][0]
            y = eigenvalue_dict[i][-1]

        li.append(np.linalg.norm(line_add_array_01[x:y, first_line[i]] - line_add_array_01[x:y, j]))
    all_li.append(min_var(li))
print(all_li)

kli = [i for i in range(209)]
for i in range(len(all_li)):
    for j in range(len(all_li[i])):
        for l in kli:
            if all_li[i][j] == l:
                kli.remove(l)
print(kli)

all_li[7].append(102)
all_li[7].append(151)
all_li[7].remove(50)
all_li[7].remove(86)

for i in range(len(all_li)):
    all_li[i].remove(first_line[i])
print(all_li)

# 行类建立tsp，每行排列完成用列表li保存
li_row = np.zeros((180, 72 * 19, 11))
for j in range(len(all_li)):
    li_num = [first_line[j]]
    imgs_recover = imgs_3d[:, :, first_line[j]]
    for k in range(len(all_li[j])):
        x = 0
        y = -1
        # 人工干预
        if j == 1 and k == (9 or 10):
            x = 0
            y = 44
        if j == 3 and k == 2:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 161]))
            li_num.append(161)
            all_li[j].remove(161)
            continue
        if j == 3 and k == 15:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 9]))
            li_num.append(9)
            all_li[j].remove(9)
            continue
        if j == 4 and k == 11:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 11]))
            li_num.append(11)
            all_li[j].remove(11)
            continue
        if j == 5 and k == 9:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 63]))
            li_num.append(63)
            all_li[j].remove(63)
            continue
        if j == 6 and k == 1:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 83]))
            li_num.append(83)
            all_li[j].remove(83)
            continue
        if j == 6 and k == 2:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 132]))
            li_num.append(132)
            all_li[j].remove(132)
            continue
        if j == 8 and k == 8:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 144]))
            li_num.append(144)
            all_li[j].remove(144)
            continue
        if j == 9 and k == 1:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 182]))
            li_num.append(182)
            all_li[j].remove(182)
            continue
        if j == 10 and k == 16:
            imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, 87]))
            li_num.append(87)
            all_li[j].remove(87)
            continue
        li = []
        for i in range(len(all_li[j])):
            li.append(np.linalg.norm(imgs_recover[x:y, -1] - imgs_3d[x:y, 0, all_li[j][i]]))
        minvar = li.index(min(li))
        imgs_recover = np.hstack((imgs_recover, imgs_3d[:, :, all_li[j][minvar]]))
        li_num.append(all_li[j][minvar])
        all_li[j].remove(all_li[j][minvar])
    print(li_num, len(li_num))
    li_row[:, :, j] = imgs_recover

imgs_recover = li_row[:, :, 0]
for i in range(1, 11):
    imgs_recover = np.vstack((imgs_recover, li_row[:, :, i]))

plt.figure(figsize=(12, 16))
plt.imshow(imgs_recover, cmap='gray')
plt.axis(False)
plt.show()

# 找出第一行


n = len(li_row[0, 0, :])
d = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x = li_row[-1, :, i]
        y = li_row[0, :, j]
        z = x - y
        d[i, j] = np.linalg.norm(z)
        if i == j:
            d[i, j] = np.inf  # 碎纸片只能和其他碎纸片相邻

###########################################################################
## 贪婪算法求解问题1
print(d)
# 先找到第一个碎纸片，第一列是空白的
for i in range(n):
    temp = li_row[0, :, i]
    if sum(temp) == 255 * len(temp):
        first = 4
        print(first)
        break

#人工干预
d[5, 10] = -np.inf
d[3, 6] = -np.inf
d[6, 10] = np.inf
d[8, 9] = -np.inf
d[0, 7] = -np.inf

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
    temp = li_row[:, :, sorted_ind[i]]
    if i == 0:
        imgs_recover = temp
    else:
        imgs_recover = np.vstack((imgs_recover, temp))

plt.figure(figsize=(12, 16))
plt.imshow(imgs_recover, cmap='gray')
plt.axis(False)
plt.show()
