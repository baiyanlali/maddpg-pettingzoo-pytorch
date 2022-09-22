import matplotlib.pyplot as plt
import numpy
# import numpy as np
import cupy as np
from PIL import Image
import time

iter_count = 8
width = 70
height = 70


kernel_size = 13
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

ipfs = np.zeros((iter_count, width, height), dtype=float)

# Agents = [[30, 20], [60, 40]]
# Landmarks = [[28, 20]]
Agents = [[30, 20], [40, 40]]
Landmarks = [[40, 20], [39, 20]]


def numpy_conv(inputs, filter):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # default np.floor
    filter_center = int(filter_size / 2.0)
    filter_center_ceil = int(np.ceil(filter_size / 2.0))

    input_new = np.zeros((H + filter_size - 1, W + filter_size - 1))
    result = np.zeros((H + filter_size - 1, W + filter_size - 1))
    # 更新下新输入,SAME模式下，会改变HW
    # H, W = inputs.shape
    # print("new size",H,W)
    # 卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置

    start = int(filter_size / 2)
    isodd = filter_size % 2 == 1
    if not isodd:
        print("error kernel not odd")
    endH = start + H
    endW = start + W

    input_new[start:endH, start:endW] = inputs

    for r in range(start, endH):
        for c in range(start, endW):
            # 池化大小的输入区域
            cur_input = input_new[r - start:r + start + 1, c - start:c + start + 1]
            # 和核进行乘法计算
            cur_output = cur_input * filter
            # 再把所有值求和
            conv_sum = np.sum(cur_output)
            # 当前点输出值
            result[r, c] = conv_sum

    return result[start:endH, start:endW]


def updateipf2():
    for agent in Agents:
        x, y = agent
        ipfs[0, x, y] = -3

    for landmark in Landmarks:
        x, y = landmark
        ipfs[0, x, y] = 5

    for i in range(len(ipfs) - 1):
        ipfs[i + 1, :, :] = numpy_conv(ipfs[i, :, :], kernel)


def updateipf():
    # update bounds to center around agent

    for agent in Agents:
        x, y = agent
        ipfs[0, x, y] = -3

    for landmark in Landmarks:
        x, y = landmark
        ipfs[0, x, y] = 5

    for i in range(len(ipfs) - 1):
        x = 0
        for y in range(1, height - 1):
            ipfs[i + 1, x, y] = (np.sum(ipfs[i, x:x + 2, y - 1:y + 2]))

        x = width - 1
        for y in range(1, width - 1):
            ipfs[i + 1, x, y] = (np.sum(ipfs[i, x - 1:x + 1, y - 1:y + 2]))  # * 0.9

        y = 0
        for x in range(1, width - 1):
            ipfs[i + 1, x, y] = (np.sum(ipfs[i, x - 1:x + 2, y:y + 2]))

        y = height - 1
        for x in range(1, width - 1):
            ipfs[i + 1, x, y] = (np.sum(ipfs[i, x - 1:x + 2, y - 1:y]))

        for x in range(1, width - 1):
            for y in range(1, width - 1):
                ipfs[i + 1, x, y] = (np.sum(ipfs[i, x - 1:x + 2, y - 1:y + 2]))  # * 0.9


def showimg(img):
    for i in range(len(ipfs) - 1):
        plt.figure(i)
        plt.axis("off")
        plt.imshow(img[i])
        plt.show()
        # img = Image.fromarray(ipfs[i])
        # img.save(f'p{i}.png')
        # img.show()  #


plt.figure(figsize=(10, 10), dpi=150)

t1 = time.time()
updateipf2()
# updateipf()

print(f"time used: {(time.time() - t1)} second")

showimg(ipfs)

# 该生自主学习能力强，对科研热情较高，基础良好，技术过硬，建议录取为专硕
