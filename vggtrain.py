import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import os
import time
from PIL import Image
import traceback


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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VGG19net(nn.Module):
    def __init__(self, num_classes=68, feature_extract=True, pretrained_weights_path='vgg19.pth'):
        super(VGG19net, self).__init__()
        # 加载预训练的 VGG19 模型
        model = models.vgg19(pretrained=False)

        # 提取 VGG19 的特征提取部分
        self.features = model.features

        # 如果进行特征提取，则冻结所有的特征提取层参数
        set_parameter_requires_grad(self.features, feature_extract)

        # 加载 VGG19 的平均池化层
        self.avgpool = model.avgpool

        # 自定义分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # 加载预训练模型权重
        self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, pretrained_weights_path):
        # 加载预训练模型权重
        state_dict = torch.load(pretrained_weights_path, map_location=device)
        self.load_state_dict(state_dict)

    def forward(self, x):
        # 通过特征提取部分
        x = self.features(x)
        # 通过平均池化层
        x = self.avgpool(x)
        # 展平操作
        x = torch.flatten(x, 1)
        # 通过自定义分类器部分
        x = self.classifier(x)
        return x
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

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=data_transform["val"])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
num_classes=68
model=VGG19net(feature_extract=True, num_classes=num_classes).to(device)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
#lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

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

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
#            scheduler.step()
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')

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
    torch.save(best_model_state_dict, 'bestVGG19.pth')