import numpy as np
import cv2
video_dir = "./data/train/MOT16-11/"
gt_path = video_dir+"gt/new_gt.txt"
gt = np.loadtxt(gt_path, delimiter=",")
Classes =[1,2,7,8,12]
pedestrainData = None
for k , c in enumerate(Classes):
    if k ==0:
        pedestrainIndex = np.where(gt[:, 7] == c)
        pedestrainData = gt[pedestrainIndex]
    else:
        pedestrainIndex = np.where(gt[:, 7] == c)
        pedestrainData_ = gt[pedestrainIndex]
        pedestrainData= np.concatenate((pedestrainData,pedestrainData_), axis=0)

pId = np.unique(pedestrainData[:,1])
for i, p in enumerate(pId):
    index = np.where(pedestrainData[:,1]==p)
    pData = pedestrainData[index]
    print(pData.shape[0])
    for j in range(pData.shape[0]):
        data = pData[j]
        imgPath = video_dir+"img1/" + '%06d' % data[0]+".jpg"
        img = cv2.imread(imgPath)
        cv2.rectangle(img, (int(data[2]),int(data[3])), (int(data[2] + data[4]), int(data[3] +data[5])), color=(255,0,0))
        cv2.putText(img, "Frame:"+str(data[0]), (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        cv2.putText(img, str(data[8]), (int(data[2]), int(data[3] - 10)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        cv2.imshow("show", img)
        cv2.waitKey(1)

