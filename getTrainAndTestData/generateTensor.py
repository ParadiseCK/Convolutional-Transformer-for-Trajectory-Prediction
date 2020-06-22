import cv2
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import glob
import os
from Kmean import getD
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


dataName = "MOT16-11"
depthImgPath = "./depthImg/"+dataName+"/depth/"
PoseDataPath = "./poseData/"+dataName+"/"
savePath = "./InitTensor/"+dataName+"/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
depthImgFilenames = sorted(glob.glob(os.path.join(depthImgPath, "*.jpg")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
PoseDataFilenames = sorted(glob.glob(os.path.join(PoseDataPath, "*.pose")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))

# Color = []
# for i in range(32):
#     Color.append(randomcolor())
# plt.ion()
# fig = plt.figure()
TENSOR = []
for num in range(len(depthImgFilenames)):
    pose = joblib.load(PoseDataFilenames[num])
    dataShape = pose.shape
    depthImg = cv2.imread(depthImgFilenames[num])
    depthImg = cv2.resize(depthImg, (dataShape[2], dataShape[1]), interpolation=cv2.INTER_NEAREST)
    depthImg = depthImg[:,:,0]
    Depth = []
    for i, data in enumerate(pose):
        d_data = data*depthImg
        index = np.where(d_data > 0)
        data_ = d_data[index]
        if data_.shape[0] > 0:
            d =getD(data_)
            Depth.append(d)
    step = 8
    TensorC =  256//step
    Tensor= np.zeros((TensorC, dataShape[1], dataShape[2]))
    for j, d in enumerate(Depth):
        d = int(d)//step
        Tensor[d,:,:] =  Tensor[d,:,:] + pose[j]
    # print("TensorShape:{}".format(Tensor.shape))
    TENSOR.append(Tensor)
    # print(num)
    # xyz = np.where(Tensor > 0)
    # x = xyz[0]
    # z = xyz[1]
    # y = xyz[2]
    #
    # ax = fig.add_subplot(111, projection='3d')
    # for m in range(len(x)):
    #     ax.scatter(x[m], y[m], 54 - z[m], c=np.array([x[m]/32, 0,0]), marker="o")
    # ax.set_xlim(0, 256 // step)
    # ax.set_ylim(0, 96)
    # ax.set_zlim(0, 54)
    # plt.show()
    # plt.pause(0.1)
    # plt.clf()
joblib.dump(np.array(TENSOR), savePath+"Tensor")