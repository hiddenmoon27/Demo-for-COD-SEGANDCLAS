import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
import time
from PIL import Image
import traceback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import torch.nn.functional as F
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集转换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 随机调整亮度、对比度、饱和度
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
# 定义一个新的模型类，继承自 ViT 模型

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
fitest_ixuanmage_paths = []
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

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=data_transform["val"])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 创建预训练模型实例
model = timm.create_model("vit_base_patch16_224", pretrained=False)

# 修改分类头为68个类别
num_classes=68
model.head = nn.Linear(model.embed_dim, num_classes)
# 加载保存的模型权重
model.load_state_dict(torch.load('bestvit21k3.pth'))


# 冷冻除了分类头以外的所有层
for name, param in model.named_parameters():
    if 'head' not in name:  # 排除分类头
        param.requires_grad = False

print(model)


model.to(device)  # 将模型移动到GPU
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=0.01, momentum=0.92, weight_decay=1E-5)
optimizer.load_state_dict(torch.load('optimizer_state_continued.pth'))

for param_group in optimizer.param_groups:
    param_group['lr'] = 0.00wodui1
criterion = nn.CrossEntropyLoss()

# 时间记录
start_time = time.time()

# 训练模型
num_epochs = 15
best_val_accuracy = 0.0  # 记录最佳验证准确率
best_model_state_dict = None  # 记录最佳模型状态字典

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_duration:.2f} seconds')

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')
        print('validation loss{}'.format(loss.item()))
        now_model=model.state_dict()
        torch.save(now_model, './m/1Epoch{}vit21k.pth'.format(epoch))
        torch.save(optimizer.state_dict(), "./m/1Epoch{}optimizer_state_continued.pth".format(epoch))
        print('保存成功')
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

total_duration = time.time() - start_time
print(f'Total Time: {total_duration:.2f} seconds')

# 保存最佳模型的状态字典
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'bestvit21k3.pth')
    torch.save(optimizer.state_dict(), "optimizer_state_continued.pth")
 #   torch.save(scheduler.state_dict(), "scheduler_state_continued.pth")