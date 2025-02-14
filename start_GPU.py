import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import copy

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 数据转换
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05)), # 增强策略已注释
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 假设您已经有了数据集路径
data_dir = "./chestxray"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 加载数据集
train_dataset = datasets.ImageFolder(train_dir, train_transforms)
val_dataset = datasets.ImageFolder(val_dir, train_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# 类别名称
class_names = train_dataset.classes

# 加载预训练模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30  # 修改为1个epoch
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs-1}')
    print('-' * 10)

    # 每个epoch包含训练和验证阶段
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 训练模式
            dataloader = train_loader
        else:
            model.eval()   # 评估模式
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # 迭代数据
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 深度复制模型权重
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:.4f}')

# 加载最佳模型权重
model.load_state_dict(best_model_wts)

# 预测函数
def predict_image(image_path, model, class_names):
    model.eval()
    image = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
    image = train_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    plt.imshow(Image.open(image_path))
    plt.title(f"Predicted: {class_names[predicted.item()]}")
    plt.axis('off')
    plt.show()

# 示例预测（使用测试集中的样本）
sample_image = "./chestxray/test/PNEUMONIA/person100_bacteria_475.jpeg"  # 修改为实际路径
predict_image(sample_image, model, class_names)

