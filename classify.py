import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch.nn as nn
Sub_Class_Dictionary = {
    'BatFish': 0, 'ClownFish': 1, 'Crab': 2, 'Crocodile': 3, 'CrocodileFish': 4, 'Fish': 5,
    'Flounder': 6, 'FrogFish': 7, 'GhostPipefish': 8, 'LeafySeaDragon': 9, 'Octopus': 10,
    'Pagurian': 11, 'Pipefish': 12, 'ScorpionFish': 13, 'SeaHorse': 14, 'Shrimp': 15, 'Slug': 16,
    'StarFish': 17, 'Stingaree': 18, 'Turtle': 19, 'Ant': 20, 'Bug': 21, 'Cat': 22, 'Caterpillar': 23,
    'Centipede': 24, 'Chameleon': 25, 'Cheetah': 26, 'Deer': 27, 'Dog': 28, 'Duck': 29, 'Gecko': 30,
    'Giraffe': 31, 'Grouse': 32, 'Human': 33, 'Kangaroo': 34, 'Leopard': 35, 'Lion': 36, 'Lizard': 37,
    'Monkey': 38, 'Rabbit': 39, 'Reccoon': 40, 'Sciuridae': 41, 'Sheep': 42, 'Snake': 43, 'Spider': 44,
    'StickInsect': 45, 'Tiger': 46, 'Wolf': 47, 'Worm': 48, 'Bat': 49, 'Bee': 50, 'Beetle': 51,
    'Bird': 52, 'Bittern': 53, 'Butterfly': 54, 'Cicada': 55, 'Dragonfly': 56, 'Frogmouth': 57,
    'Grasshopper': 58, 'Heron': 59, 'Katydid': 60, 'Mantis': 61, 'Mockingbird': 62, 'Moth': 63,
    'Owl': 64, 'Owlfly': 65, 'Frog': 66, 'Toad': 67
}
def get_key_from_value(value):
    d=Sub_Class_Dictionary
    keys = [k for k, v in d.items() if v == value]
    return keys[0] if keys else None
def classify(image_path):
    #这里运用的是vitb16
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批量维度

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: 创建与保存模型结构相同的模型
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    num_classes = 68  # 设置你的分类头
    model.head = nn.Linear(model.embed_dim, num_classes)
    model.load_state_dict(torch.load("Epoch5vit21k.pth"))
    model.to(device)  # 将模型移动到GPU
    # 将图片移到与模型相同的设备
    image = image.to(device)
    model.eval()  # 设置模型为评估模式
    # 禁用梯度计算（加快推理速度并降低内存消耗）
    with torch.no_grad():
        output = model(image)
    # 对输出进行 softmax 处理
    probabilities = F.softmax(output[0], dim=0)

    # 获取概率最高的类别索引
    predicted_class_index = torch.argmax(probabilities).item()
    result=predicted_class_index
    # 打印预测结果
    print("预测类别索引:", predicted_class_index)

    return get_key_from_value(result)
def classify1(image_path):
    #这里运用的是vgg19
    return image_path

if __name__ == '__main__':
    print(classify('BatFish.png') )