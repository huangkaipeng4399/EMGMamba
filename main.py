import os
import numpy as np
from mamba import Mamba, ModelArgs
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import logging
import sys
import math

import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EMGDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.from_numpy(self.x[idx]), torch.tensor(
                self.y[idx], dtype=torch.long)
        else:
            return torch.from_numpy(self.x[idx])


import torch
import torch.nn as nn

# class FaceMAELoss(nn.Module):
#     def __init__(self, w=20, epsilon=1e-5):
#         super(FaceMAELoss, self).__init__()
#         self.w = w
#         self.epsilon = epsilon

#     def forward(self, pred_keypoints, true_keypoints):
#         # 计算欧式距离
#         # print(pred_keypoints.shape)
#         # print(true_keypoints.shape)
#         pred_keypoints=pred_keypoints.reshape(32,51,2)
#         distance = torch.norm(pred_keypoints - true_keypoints, dim=-1)  # dim=-1 表示在最后一个维度上计算欧式距离

#         # 根据距离计算损失
#         loss = torch.where(distance < self.w,
#                            self.w * torch.log(1 + distance / self.epsilon),
#                            distance - (self.w - self.w * torch.log(1 + self.w / self.epsilon)))

#         # 求平均损失
#         loss = torch.mean(loss)

#         return loss

if __name__ == '__main__':
    torch.cuda.set_device(1)
    muscle_data_train = np.load(
        "/user/ZY/EMG2Face/EMG3D/data/data_align/align_emg_data_train.npy",
        allow_pickle=True)  # 导入训练数据集

    landmarks_train = np.load(
        "/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_train.npy",
        allow_pickle=True)
    # muscle_data_train=torch.tensor(muscle_data_train)
    # landmarks_train=torch.tensor(landmarks_train)

    train_dataset = EMGDataset(muscle_data_train, landmarks_train)
    dataloader = DataLoader(train_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=1)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #input_size:[b,]
    model_args = ModelArgs(
        emg_features=2,
        d_model=16,  #TBD
        n_layer=2,  #TBD
        vocab_size=102,  #return size 
    )
    model = Mamba(model_args).to('cuda')
    print(model)

    loss_model = nn.L1Loss()

    learning_rate = 0.01
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练的轮数
    epoches = 40
    model.train()  # prep model for training
    logging.info(f'GPU is available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        logging.info(f"Train model on {gpu_num} GPUs:")
        for i in range(gpu_num):
            print('\t GPU {}.: {}'.format(i, torch.cuda.get_device_name(i)))
        model = model.cuda()
    trainlosslist = []

    for epoch in range(epoches):
        # monitor training loss
        train_loss = 0.0
        i = 0
        out_list = []  #debug

        for data, label in dataloader:
            i += 1
            data = data.cuda()
            label = label.cuda()
            outputs = model(data)
            # raise SystemExit("after this")
            # outputs=outputs.reshape(32,51,2)
            #丢弃剩余的样本
            optimizer_model.zero_grad()
            outputs = outputs.to(torch.float32)
            label = label.to(torch.float32)
            loss = loss_model(outputs, label)
            # print(loss)
            loss.backward()
            out_list.append(outputs[0, 0, :])
            if i % 50 == 0:
                logging.info(f"Loss: {loss.item()}")
            if math.isnan(loss.item()):
                print("nan!")
                print(out_list)
                print(outputs)
                print(torch.isfinite(outputs))
                print(torch.isfinite(label))
                sys.exit()

                # print(f"Loss: {loss.item()}")
            train_loss += loss.item()
            optimizer_model.step()
        train_loss = train_loss / len(dataloader.dataset)
        trainlosslist.append(train_loss)
        logging.info(f"Epoch : {epoch}, traning loss:{train_loss}")

    epoch_idx = [i for i in range(1, epoches + 1)]
    # 创建图形窗口

    # 绘制折线图
    try:
        plt.figure()
        plt.plot(epoch_idx, trainlosslist)

        # 添加标签和标题
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train_loss')
        plt.savefig(f'e{epoches}d{model_args.d_model}lay{model_args.n_layer}.png')
    except:
        logging.error("fail to plt")
    torch.save(model, f'e{epoches}d{model_args.d_model}lay{model_args.n_layer}.pth')
