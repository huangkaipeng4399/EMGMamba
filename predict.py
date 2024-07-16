import torch
import numpy as np
import matplotlib.pyplot as plt

model = torch.load("model0702.pth")
model.eval()

muscle_data_test = np.load(
    "/user/ZY/EMG2Face/EMG3D/data/data_align/align_emg_data_test.npy",
    allow_pickle=True)
input_tensor = torch.tensor(muscle_data_test).to("cuda:1")

with torch.no_grad():
    predictions = model(input_tensor)

landmarks_test = np.load(
    "/user/ZY/EMG2Face/EMG3D/data/data_new51/face_points_test.npy",
    allow_pickle=True)
# predictions=landmarks_train

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
x = [prediction[0] for prediction in predictions[110].cpu().numpy()]
y = [-prediction[1] for prediction in predictions[110].cpu().numpy()]

# 设置图表标题和坐标轴标签
axs[0].scatter(x, y)
axs[0].set_title('EMGMamba predict')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')

x_gt = [i[0] for i in landmarks_test[110]]
y_gt = [-i[1] for i in landmarks_test[110]]
axs[1].scatter(x, y)
axs[1].set_title("ground truth")
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
# 显示图表

plt.savefig("prediction2.png")
