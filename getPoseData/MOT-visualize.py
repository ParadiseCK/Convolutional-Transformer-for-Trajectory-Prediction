import numpy as np
import random
import glob
import os
import cv2
def random_color(num):
    COLOR=[]
    for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        COLOR.append((r,g,b))
    return COLOR
video_dir = "./data/train/MOT16-11/"
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
for i, p in enumerate(imgames):
    frameAll = gt[:,0]
    index = np.where(frameAll == i+1)
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
    for j, data in enumerate(pedestrainData):
        pid = int(data[1])
        co = color[pid-1]
        x = int(data[2])
        y = int(data[3])
        x_ = int(data[2]+data[4])
        y_ = int(data[3]+data[5])
        if data[8] > 0.6:
            cv2.rectangle(img, (x, y),(x_,y_), co,2)
            cv2.putText(img, str(ClassNames[int(data[7])-1])+"_"+str(pid), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,co, 1)
    cv2.imshow("show", img)
    cv2.waitKey(1)
