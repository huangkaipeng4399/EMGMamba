import os
import numpy as np
from mamba import Mamba,ModelArgs
import torch
import torch.nn as nn

class face_mae(nn.Module):
    def __init__(self) -> None:
        super(face_mae,self).__init__()
    
    def forward(self):
        return



if __name__=='__main__':
    muscle_data_train = np.load("/user/ZY/EMG2Face/EMG3D/data/data_align/align_emg_data_train.npy", allow_pickle=True)  # 导入训练数据集
    
    landmarks_train = np.load("/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_train.npy", allow_pickle=True) 
    train_data=torch.from_numpy(landmarks_train)
    model_args=ModelArgs(
        d_model=1,
        n_layer=5,  #TBD
        vocab_size=102  #return size 



    )
    model=Mamba(model_args).to('cuda')
    print(model)
    
