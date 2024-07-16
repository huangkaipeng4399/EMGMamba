import os
import numpy as np
import matplotlib.pyplot as plt

landmarks_train = np.load("/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_train.npy", allow_pickle=True) 
landmarks_test = np.load("/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_test.npy", allow_pickle=True)
landmarks_val = np.load("/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_val.npy", allow_pickle=True) 

muscle_data_train = np.load("/user/ZY/EMG2Face/EMG3D/data/data_align/align_emg_data_train.npy", allow_pickle=True)
print(muscle_data_train.shape)
print(landmarks_train.shape)
