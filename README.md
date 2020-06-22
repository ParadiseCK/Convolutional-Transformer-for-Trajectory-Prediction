# Introduction
This is the code implementation of 'Future Pedestrian Location Prediction in Egocentric Videos for Autonomous Vehicles'
# Prerequisites  
python 3.6  
ubuntu 14.04  
cuda-8.0  
cudnn-6.0.21  
Pytorch-1.0.1-gpu  
NVIDIA GTX 1080Ti  
# Prepare img depth 
cd getDepth
python run_mot.py
python getDepth.py
# Prepare pose data
cd getPoseData
python getPose.py
# Combining the img depth,  the pedestrian pose and the xy-coordinates to construct the multi-channel tensor
cd getTranAndTestData
python generateNewGt.py
python generateTensor.py
python generateTrainData.py
# Train
cd Conv-Transformer
python train --batchSize --epochs --lrDecay --c_model --N --heads --s_model --lr
# Test
cd Conv-Transformer
python runPredict.py
