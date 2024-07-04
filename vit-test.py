import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
import os
import time
from PIL import Image
import traceback
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
test_dataframe = pd.read_excel('fittest.xlsx', usecols=[0, 1])  # 读取指定列
image_folder = './t1'
image_paths = [os.path.join(image_folder, fname) for fname in test_dataframe['Original Filename']]
# 提取标签信息，即 traintag 的 Mapped Label 列
labels = test_dataframe['Mapped Label'].tolist()
test_dataset = CustomDataset(image_paths, labels, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 时间记录
# 初始化模型
num_classes = 68
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('bestvits16.pth', map_location=device))
model.to(device)  # 将模型移动到GPU
model.eval()
start_time = time.time()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total
finish_time=time.time()
print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')
print(f'{finish_time-start_time}')