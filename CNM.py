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
import cv2
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_dataframe = pd.read_excel('newtesttag.xlsx', usecols=[0, 1])  # 读取指定列
image_folder = './t1'
image_paths = [os.path.join(image_folder, fname) for fname in test_dataframe['Original Filename']]
# 提取标签信息，即 traintag 的 Mapped Label 列
labels = test_dataframe['Mapped Label'].tolist()
test_dataset = CustomDataset(image_paths, labels, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 时间记录
# 初始化模型
num_classes = 68
model = timm.create_model('vit_small_patch16_224', pretrained=False)
model.head = torch.nn.Linear(model.head.in_features, num_classes)
model.load_state_dict(torch.load('vit_S16.pt', map_location=device))
model.to(device)  # 将模型移动到GPU
model.eval()
start_time = time.time()
correct = 0
total = 0
count=1
print("ready")
with torch.no_grad():
    for images, labels in test_loader:
        print(count)
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count=count+1

val_accuracy = correct / total
finish_time=time.time()
print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')
print(f'{finish_time-start_time}')
#
# from maketag import get_substring_between_last_two_underscores,save_labels_to_excel
# import pandas as pd
# k=['beetle', 'spider', 'chameleon', 'wolf', 'turtle', 'cheetah', 'snake', 'katydid', 'pipefish', 'reccoon', 'ant', 'butterfly', 'lion', 'stickInsect', 'grasshopper', 'flounder', 'dog', 'toad', 'owlfly', 'gecko', 'bee', 'bird', 'owl', 'caterpillar', 'leafySeaDragon', 'cicada', 'stingaree', 'tiger', 'lizard', 'heron', 'ghostPipefish', 'bat', 'kangaroo', 'starFish', 'frogFish', 'worm', 'clownFish', 'crocodileFish', 'bug', 'frog', 'sheep', 'dragonfly', 'octopus', 'grouse', 'mantis', 'shrimp', 'fish', 'pagurian', 'duck', 'batFish', 'rabbit', 'giraffe', 'crab', 'monkey', 'seaHorse', 'centipede', 'scorpionFish', 'moth', 'leopard', 'crocodile', 'bittern', 'sciuridae', 'cat', 'deer', 'mockingbird', 'slug', 'frogmouth', 'human']
# Sub_Class_Dictionary = {
#     'BatFish': 0, 'ClownFish': 1, 'Crab': 2, 'Crocodile': 3, 'CrocodileFish': 4, 'Fish': 5,
#     'Flounder': 6, 'FrogFish': 7, 'GhostPipefish': 8, 'LeafySeaDragon': 9, 'Octopus': 10,
#     'Pagurian': 11, 'Pipefish': 12, 'ScorpionFish': 13, 'SeaHorse': 14, 'Shrimp': 15, 'Slug': 16,
#     'StarFish': 17, 'Stingaree': 18, 'Turtle': 19, 'Ant': 20, 'Bug': 21, 'Cat': 22, 'Caterpillar': 23,
#     'Centipede': 24, 'Chameleon': 25, 'Cheetah': 26, 'Deer': 27, 'Dog': 28, 'Duck': 29, 'Gecko': 30,
#     'Giraffe': 31, 'Grouse': 32, 'Human': 33, 'Kangaroo': 34, 'Leopard': 35, 'Lion': 36, 'Lizard': 37,
#     'Monkey': 38, 'Rabbit': 39, 'Reccoon': 40, 'Sciuridae': 41, 'Sheep': 42, 'Snake': 43, 'Spider': 44,
#     'StickInsect': 45, 'Tiger': 46, 'Wolf': 47, 'Worm': 48, 'Bat': 49, 'Bee': 50, 'Beetle': 51,
#     'Bird': 52, 'Bittern': 53, 'Butterfly': 54, 'Cicada': 55, 'Dragonfly': 56, 'Frogmouth': 57,
#     'Grasshopper': 58, 'Heron': 59, 'Katydid': 60, 'Mantis': 61, 'Mockingbird': 62, 'Moth': 63,
#     'Owl': 64, 'Owlfly': 65, 'Frog': 66, 'Toad': 67
# }
# lowercase_dict = {key.lower(): value for key, value in Sub_Class_Dictionary.items()}
# lowercase_list = [item.lower() for item in k]
#
#
#
# # 打开 Excel 文件
# file_path = 'testtag.xlsx'  # 替换为你的文件路径
# df = pd.read_excel(file_path)
#
# if 'Original Filename' in df.columns and 'Mapped Label' in df.columns:
#     # 选择这两列的数据
#     filenames = df['Original Filename'].tolist()
#     labels = df['Mapped Label'].tolist()
# lower_filename=[item.lower() for item in filenames]
# count=0
# for i in lower_filename:
#     ii=get_substring_between_last_two_underscores(i)
#     dex=lowercase_list.index(ii)
#     labels[count]=dex
#     count=count+1
# print(min(labels),max(labels))
# path='newtesttag.xlsx'
# save_labels_to_excel(labels,filenames,path)














