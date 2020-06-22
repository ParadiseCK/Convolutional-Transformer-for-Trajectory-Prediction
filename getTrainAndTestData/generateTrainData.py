import numpy as np
import cv2
from sklearn.externals import joblib
import os
video_dir = "./data/train/MOT16-11/"
depthImg_dir = "./depthImg/MOT16-11/depth/"
gt_path = video_dir+"gt/new_gt.txt"
gt = np.loadtxt(gt_path, delimiter=",")
initTensor = joblib.load("./InitTensor/MOT16-11/Tensor")
SavePath = "./datasets/MOT16-11/"
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
Train = []
Label = []
pId = np.unique(gt[:,1])
for i, p in enumerate(pId):
    index = np.where(gt[:,1]==p)
    pData = gt[index]
    PedestrianTensor = []
    PedestrianDepth =[]
    PedestrianX = []
    PedestrianY = []
    for j in range(pData.shape[0]):
        gtdata = pData[j]
        frame = int(gtdata[0]-1)
        depthImgPath = depthImg_dir + str(frame+1).zfill(6)+".jpg"
        depthImg = cv2.imread(depthImgPath)
        depthImg = cv2.resize(depthImg, (96, 54), interpolation=cv2.INTER_NEAREST)
        depthImg_ = depthImg[:, :, 0]
        PedestrianX.append((gtdata[2] + gtdata[4] / 2) / 1920)
        PedestrianY.append((gtdata[3] + gtdata[5] / 2) / 1080)
        x = gtdata[2]//20
        y = gtdata[3]//20
        x_ = (gtdata[2] + gtdata[4])//20
        y_ = (gtdata[3] + gtdata[5])//20
        c_x = int((x+x_)/2)
        c_y = int((y+y_)/2)
        # c_x_1 = c_x - 2
        # c_y_1 = c_y - 2
        # c_x_2 = c_x + 2
        # c_y_2 = c_y + 2
        if c_x >=96:
            c_x=96-1
        if c_y >=54:
            c_y=54-1
        # cv2.rectangle(depthImg, (int(c_x_1), int(c_y_1)), (int(c_x_2), int(c_y_2)),
        #               color=(0, 0, 255))
        # cv2.circle(depthImg, (c_x, c_y),4, (0, 0, 255), 0 )
        # cv2.imshow("111", depthImg)
        # cv2.waitKey(1)
        # p_depth = np.mean(depthImg_[c_y_1:c_y_2,c_x_1:c_x_2])//8
        p_depth =depthImg_[c_y][c_x]// 8
        ration = gtdata[8]
        init_tensor = initTensor[frame]*-1
        if ration >= 0.5:
        # print(np.sum(init_tensor))
            pose_index = np.where(init_tensor!=0)
            d_index = pose_index[0]
            x_index = pose_index[2]
            y_index = pose_index[1]
            d_index_ = []
            x_index_ = []
            y_index_ = []
            for l in range(d_index.shape[0]):
                d = d_index[l]
                x__ = x_index[l]
                y__ = y_index[l]
                if x<x__<x_ and y<y__<y_:
                    d_index_.append(d)
                    x_index_.append(x__)
                    y_index_.append(y__)
            if len(d_index_)>0:
                d_index_ = np.array(d_index_)
                x_index_ = np.array(x_index_)
                y_index_ = np.array(y_index_)
                # print(p_depth)
                # print(d_index_)
                depth_ = np.ones(d_index_.shape[0])*p_depth
                gapArr = abs(depth_-d_index_)
                gap_ = np.min(gapArr)
                gap_index = np.where(gapArr==gap_)

                d_index_ = d_index_[gap_index]
                x_index_ = x_index_[gap_index]
                y_index_ = y_index_[gap_index]
                pose_index_ =(d_index_,  y_index_, x_index_)
                # print(pose_index_[0])
                # print("****"*10)
                init_tensor[pose_index_] = 1
                p_depth = d_index_[0]
        PedestrianTensor.append(init_tensor)
        PedestrianDepth.append(p_depth/32)
    PedestrianTensor = np.array(PedestrianTensor)
    PedestrianDepth = np.array(PedestrianDepth)
    PedestrianX = np.array(PedestrianX)
    PedestrianY = np.array(PedestrianY)

    for k in range(PedestrianTensor.shape[0]):
        if k +20 < PedestrianTensor.shape[0]:
            train = PedestrianTensor[k:k+7]
            D = PedestrianDepth[k+7:k+19]
            X = PedestrianX[k + 7:k + 19]
            Y = PedestrianY[k + 7:k + 19]
            D_ = PedestrianDepth[k + 8:k + 20]
            X_ = PedestrianX[k + 8:k + 20]
            Y_ = PedestrianY[k + 8:k + 20]
            label_in = []
            label_out = []
            for n in range(D.shape[0]):
                label_in.append([[[D[n]]], [[X[n]]], [[Y[n]]]])
                label_out.append([[[D_[n]]], [[X_[n]]], [[Y_[n]]]])
            lable = [np.array(label_in), np.array(label_out)]
            lable = np.array(lable)
            Train.append(train)
            Label.append(lable)
Train =  np.array(Train, dtype="float32")
Label = np.array(Label, dtype="float32")
joblib.dump(Train, SavePath+"train")
joblib.dump(Label, SavePath+"label")



