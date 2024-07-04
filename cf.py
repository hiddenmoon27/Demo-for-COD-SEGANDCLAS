
# import torch
# import torch.nn as nn
# import timm
#
# # 创建预训练的ViT模型
#
# model = timm.create_model("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
#
# # 打印模型结构
# print(model)
#
# # 添加分类头
# num_classes = 68  # 修改为你的类别数量
# model.head = nn.Linear(model.embed_dim, num_classes)
#
# # 冻结除了分类头之外的所有层
# for name, param in model.named_parameters():
#     if "head" not in name:
#         param.requires_grad = False
#
# # 打印添加分类头后的模型结构
# print(model)
#
# # 保存模型
# torch.save(model.state_dict(), "vits16.pth")
#
# # 载入保存的模型
# model.load_state_dict(torch.load("vits16.pth"))
#
# # 检查分类头是否正确
# print(model.head)

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
#
# path='newtesttag.xlsx'
# save_labels_to_excel(labels,filenames,path)

print("Epoch 100 :Loss: 0.7227, Time: 217.01 seconds Validation Accuracy: 65.08% \nvalidation loss0.9196333289146423")




