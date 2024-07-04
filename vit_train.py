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
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch.nn.functional as F
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集转换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( [0.5,0.5,0.5],  [0.5,0.5,0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( [0.5,0.5,0.5],  [0.5,0.5,0.5])
    ])
}

# 自定义数据集类
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

# 读取 traintag.xlsx 文件
label_dataframe = pd.read_excel('traintag.xlsx', usecols=[0, 1])  # 读取指定列
test_dataframe = pd.read_excel('testtag.xlsx', usecols=[0, 1])  # 读取指定列

# 构建 image_paths 列表，导入存在于 traintag 的 Original Filename 列中的图片文件路径
image_folder = './tr'
image_paths = [os.path.join(image_folder, fname) for fname in label_dataframe['Original Filename']]

# 提取标签信息，即 traintag 的 Mapped Label 列
labels = label_dataframe['Mapped Label'].tolist()

# 提取 testtag 中的验证集
test_image_paths = [os.path.join('./t1', fname) for fname in test_dataframe['Original Filename']]
test_labels = test_dataframe['Mapped Label'].tolist()

# 创建自定义数据集
train_dataset = CustomDataset(image_paths, labels, transform=data_transform["train"])

# 创建验证集
val_image_paths = []
val_labels = []
fitest_image_paths = []
fitest_labels = []

for label in range(68):
    indices = [i for i, x in enumerate(test_labels) if x == label]
    num_samples = len(indices)
    if num_samples == 0:
        continue
    if num_samples % 2 == 0:
        val_indices = indices[:num_samples // 2]
    else:
        val_indices = indices[:(num_samples - 1) // 2]

    val_image_paths.extend([test_image_paths[i] for i in val_indices])
    val_labels.extend([test_labels[i] for i in val_indices])

    fitest_indices = indices[len(val_indices):]
    fitest_image_paths.extend([test_image_paths[i] for i in fitest_indices])
    fitest_labels.extend([test_labels[i] for i in fitest_indices])

# 写入 fitest.xlsx
fitest_df = pd.DataFrame({
    'Original Filename': [os.path.basename(path) for path in fitest_image_paths],
    'Mapped Label': fitest_labels
})
fitest_df.to_excel('fitest.xlsx', index=False)


# 定义一个新的模型类，继承自 ViT 模型
class ViTWithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super(ViTWithDropout, self).__init__()
        self.original_model = original_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.original_model(x)  # 这里直接调用 ViT 模型，得到 [batch_size, num_classes] 的输出
        x = self.dropout(x)  # 在输出后应用 dropout
        return x


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=data_transform["val"])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
num_classes = 68
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('vitb16.pth', map_location=device))
model.to(device)  # 将模型移动到GPU

# 用 dropout 包装原始模型
dropout_rate = 0.9  # 你可以根据需要调整 dropout 率
model_with_dropout = ViTWithDropout(model, dropout_rate)
model_with_dropout.to(device)
for name, param in model_with_dropout.named_parameters():
    if not name.startswith("original_model.head"):  # 如果参数名称不以 "head" 开头，就冻结它
        param.requires_grad_(False)

pg = [p for p in model_with_dropout.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

criterion = nn.CrossEntropyLoss()

# 时间记录
start_time = time.time()

# 训练模型
num_epochs = 100
best_val_accuracy = 0.0  # 记录最佳验证准确率
best_model_state_dict = None  # 记录最佳模型状态字典

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model_with_dropout.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = model_with_dropout(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_duration:.2f} seconds')

        # 验证模型
        model_with_dropout.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
                outputs = model_with_dropout(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model_with_dropout.state_dict()

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

total_duration = time.time() - start_time
print(f'Total Time: {total_duration:.2f} seconds')

# 保存最佳模型的状态字典
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, '1bestvitb16.pth')



