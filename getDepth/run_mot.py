import glob
import os

video_dir = "./data/train/MOT16-13/"
filenames = sorted(glob.glob(os.path.join(video_dir, "img1/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))

txtName = "./data/test_list.txt"
f=open(txtName,'w')
for i, path in enumerate(filenames):
    f.write(path+"\n")
f.close()