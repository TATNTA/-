import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# 数据路径
data_dir = 'data/'  # 假设数据集在这个文件夹中
train_file = 'train.txt'
test_file = 'test_without_label.txt'

# 读取train.txt
train_df = pd.read_csv(train_file, names=['guid', 'tag'])
test_df = pd.read_csv(test_file, names=['guid', 'tag'])

# 打印原始标签
print("原始标签：", train_df['tag'].unique())

# 过滤无效标签，只保留 valid_tags 中的标签
valid_tags = ['positive', 'neutral', 'negative']
train_df = train_df[train_df['tag'].isin(valid_tags)]  # 只保留有效标签

# 打印过滤后的标签
print("过滤后的标签：", train_df['tag'].unique())

# 标签编码，确保标签值在 0, 1, 2 范围内
label_encoder = LabelEncoder()
train_df['tag_encoded'] = label_encoder.fit_transform(train_df['tag'])

# 打印标签编码后的唯一值，确保值为 [0, 1, 2]
print("标签编码后的唯一值:", train_df['tag_encoded'].unique())  # 检查标签编码后的唯一值

# 强制确保标签只在 0, 1, 2 范围内
assert all(train_df['tag_encoded'].isin([0, 1, 2])), "标签值超出范围"

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_df['guid'], train_df['tag_encoded'], test_size=0.2, random_state=42)

# 自定义Dataset类，加载文本和图像
class MultimodalDataset(Dataset):
    def __init__(self, guids, labels, data_dir, tokenizer, transform=None, max_length=128):
        self.guids = guids
        self.labels = labels
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx):
        guid = self.guids[idx]
        label = self.labels[idx]

        # 确保正确拼接文本文件路径
        text_path = os.path.join(self.data_dir, f"{guid}.txt")

        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except FileNotFoundError:
            text = ""  # 如果文件不存在，返回空字符串

        # 文本编码，确保padding和截断
        text_input = self.tokenizer(text,
                                    padding='max_length',  # 填充至最大长度
                                    truncation=True,  # 截断超长文本
                                    max_length=self.max_length,  # 设置最大长度
                                    return_tensors='pt')

        # 确保正确拼接图像文件路径
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224))  # 如果图像文件不存在，返回一个空白图像

        if self.transform:
            image = self.transform(image)

        return text_input, image, label


# 设置BERT模型和图像预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据加载器
train_dataset = MultimodalDataset(X_train.values, y_train.values, data_dir, tokenizer, transform=image_transform,
                                  max_length=128)
val_dataset = MultimodalDataset(X_val.values, y_val.values, data_dir, tokenizer, transform=image_transform,
                                max_length=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型设计（多模态融合模型）
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=3, use_text=True, use_image=True):
        super(MultimodalFusionModel, self).__init__()

        # 文本模型部分（BERT）
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 图像模型部分（ResNet）
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 不需要最后的分类层

        # 使用标志位来决定是否使用文本和图像
        self.use_text = use_text
        self.use_image = use_image

        # 融合层
        input_size = 0
        if self.use_text:
            input_size += self.bert.config.hidden_size
        if self.use_image:
            input_size += 2048  # ResNet 的输出特征维度

        self.fc = nn.Linear(input_size, num_classes)  # BERT hidden_size + ResNet features

    def forward(self, text_input, image):
        features = []

        # BERT文本特征
        if self.use_text:
            text_output = self.bert(**text_input)
            text_features = text_output.pooler_output
            features.append(text_features)

        # ResNet图像特征
        if self.use_image:
            image_features = self.resnet(image)
            features.append(image_features)

        # 特征融合
        combined_features = torch.cat(features, dim=1)  # 将选中的特征拼接在一起

        # 分类层
        output = self.fc(combined_features)
        return output


# 实例化模型
multimodal_model = MultimodalFusionModel(num_classes=3, use_text=True, use_image=True)
multimodal_model = multimodal_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(multimodal_model.parameters(), lr=5e-6)

# 训练模型，增加早停机制
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3, patience=2):
    best_val_loss = float('inf')  # 初始化为无穷大
    patience_counter = 0  # 用于计数在验证集上没有改进的轮次

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for text_input, image, label in tqdm(train_loader):
            text_input = {key: val.squeeze(1).to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in text_input.items()}
            image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
            label = label.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            output = model(text_input, image)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(output, 1)
            correct_predictions += torch.sum(preds == label)
            total_predictions += label.size(0)

        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for text_input, image, label in tqdm(val_loader):
                text_input = {key: val.squeeze(1).to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in text_input.items()}
                image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
                label = label.to('cuda' if torch.cuda.is_available() else 'cpu')

                output = model(text_input, image)
                loss = criterion(output, label)
                val_loss += loss.item()

                _, preds = torch.max(output, 1)
                correct_predictions += torch.sum(preds == label)
                total_predictions += label.size(0)

        val_accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 如果验证损失改善，重置计数器
        else:
            patience_counter += 1

        # 如果验证损失没有改善超过`patience`个周期，提前停止
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# 训练模型
train_model(multimodal_model, train_loader, val_loader, criterion, optimizer, epochs=2)

# 进行消融实验，分别训练单模态模型
# 训练文本模型
text_only_model = MultimodalFusionModel(num_classes=3, use_text=True, use_image=False)
text_only_model = text_only_model.to('cuda' if torch.cuda.is_available() else 'cpu')
train_model(text_only_model, train_loader, val_loader, criterion, optimizer, epochs=2)

# 训练图像模型
image_only_model = MultimodalFusionModel(num_classes=3, use_text=False, use_image=True)
image_only_model = image_only_model.to('cuda' if torch.cuda.is_available() else 'cpu')
train_model(image_only_model, train_loader, val_loader, criterion, optimizer, epochs=2)
