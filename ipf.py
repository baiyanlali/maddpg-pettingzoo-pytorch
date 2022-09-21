import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

iter_count = 50
width = 50
height = 50

ipfs = np.zeros((iter_count, width, height), dtype=float)

# Agents = [[30, 20], [60, 40]]
# Landmarks = [[28, 20]]
Agents = [[30, 20], [40, 40]]
Landmarks = [[40, 20], [39, 20]]


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
                ipfs[i + 1, x, y] = (np.sum(ipfs[i, x - 1:x + 2, y - 1:y + 2])) #* 0.9


def showimg(img):
    for i in range(len(ipfs) - 1):
        plt.figure(i)
        plt.axis("off")
        plt.imshow(img[i])
        plt.show()
        # img = Image.fromarray(ipfs[i])
        # img.save(f'p{i}.png')
        # img.show()  #
plt.figure(figsize=(10,10),dpi=150)

updateipf()
showimg(ipfs)
pass

#该生自主学习能力强，对科研热情较高，基础良好，技术过硬，建议录取为专硕