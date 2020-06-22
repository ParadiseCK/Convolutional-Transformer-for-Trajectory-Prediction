import numpy as np
import random
import glob
import os
import cv2
from sklearn.externals import joblib
from pose.src.body import *
def getPose(subset):
    X = []
    Y = []
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                X.append(-1)
                Y.append(-1)
                continue
            x, y = candidate[index][0:2]
            X.append(x)
            # Y.append(1080-y)
            Y.append(y)
    X_ = []
    Y_ = []
    for i in range(len(X) // 18):
        x = []
        y = []
        for j in range(18):
            index = j * (len(X) // 18) + i
            # print(index)
            x_, y_ = X[index], Y[index]
            if x_ != -1 and y_ != -1:
                x.append(x_)
                y.append(y_)
        X_.append(np.array(x))
        Y_.append(np.array(y))
    X_ = np.array(X_)
    Y_ = np.array(Y_)
    return X_, Y_
def random_color(num):
    COLOR=[]
    for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        COLOR.append((r,g,b))
    return COLOR
video_dir = "./data/train/MOT16-11/"
savePath = "./PoseData/MOT16-11/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
gt_path = video_dir+"gt/gt.txt"
Classes =[1,2,7]
ClassNames = ["Pedestrian","Person on vehicle", "car", "Bicycle", "Motorbike",
              "Non motorized vehicle", "Static person", "Distractor", "Occluder",
              "Occuluder on the gruond", "Occuluder full", "Reflection"]
gt = np.loadtxt(gt_path, delimiter=",")
pId = np.unique(gt[:,1])
totalNum = int(np.max(pId))
color = random_color(totalNum)
imgames = sorted(glob.glob(os.path.join(video_dir, "img1/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
body_estimation = Body('./pose/model/body_pose_model.pth')
for num, p in enumerate(imgames):
    frameAll = gt[:,0]
    index = np.where(frameAll == num+1)
    frameData = gt[index]
    pedestrainData = None
    for k , c in enumerate(Classes):
        if k ==0:
            pedestrainIndex = np.where(frameData[:, 7] == c)
            pedestrainData = frameData[pedestrainIndex]
        else:
            pedestrainIndex = np.where(frameData[:, 7] == c)
            pedestrainData_ = frameData[pedestrainIndex]
            pedestrainData= np.concatenate((pedestrainData,pedestrainData_), axis=0)
    img = cv2.imread(p)
    img_X = []
    img_Y = []
    for j, data in enumerate(pedestrainData):
        pid = int(data[1])
        co = color[pid-1]
        x = int(data[2])
        y = int(data[3])
        x_ = int(data[2]+data[4])
        y_ = int(data[3]+data[5])
        if data[8] > 0.5:
            if x <=0:
                x= 0
            if y <=0:
                y= 0
            if x_ <=0:
                x_= 0
            if y_ <=0:
                y_= 0
            oriImg = img[y:y_, x:x_]
            w = oriImg.shape[1]
            h = oriImg.shape[0]
            candidate, subset = body_estimation(oriImg)
            X, Y = getPose(subset)
            if X.shape[0] > 0 and Y.shape[0] >0:
                X , Y = X[0], Y[0]
                # print(X.shape)
                # for o in range(X.shape[0]):
                #     cv2.circle(oriImg, (int(X[o]), int(Y[o])), 5, color[j], -1)
                # cv2.imshow("iii", oriImg)
                # cv2.waitKey(1000)
                X_ = []
                Y_ = []
                for d in X:
                    X_.append(d+ x)
                for d in Y:
                    Y_.append(d+y)
                # for o in range(len(X_)):
                #     cv2.circle(img, (int(X_[o]), int(Y_[o])), 5, color[j], -1)

                img_X.append(X_)
                img_Y.append(Y_)
    # cv2.imshow("111", img)
    # cv2.waitKey(0)

    img_X = np.array(img_X)
    img_Y = np.array(img_Y)
    DATA = []
    for k in range(img_X.shape[0]):
        data = np.zeros((img.shape[0], img.shape[1]))
        for i in range(len(img_X[k])):
            x = int(img_X[k][i])
            y = int(img_Y[k][i])
            r = 10
            for i in range(-r, r, 1):
                for j in range(-r, r, 1):
                    if 0 < (y + i) < 1080 and 0 < (x + j) < 1920:
                        data[y + i][x + j] = 1
        data = cv2.resize(data, (96, 54), interpolation=cv2.INTER_NEAREST)
        DATA.append(data)
        # plt.imshow(data)
        # plt.show()
    DATA = np.array(DATA)
    num_ = num + 1
    joblib.dump(DATA, savePath + str(num_).zfill(6) + ".pose")
    print("GeneratePose:{}".format(str(num_).zfill(6) + ".pose with size:{}".format(DATA.shape)))









