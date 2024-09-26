import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 二、神经网络
# 1. 数据加载
data = pd.read_csv('../data/train.csv')
X_raw = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_raw.info()
# %%
# 2. 数据预处理（填充缺失值、标准化、独热编码）
# 数值数据
numeric_features = X_raw.select_dtypes(include='number').columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# 类别数据
categorical_features = X_raw.select_dtypes(exclude='number').columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value="unknown")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = preprocessor.fit_transform(X_raw).toarray()
# %%
# 3. 数据集分隔
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
# %%
# 4. 神经网络定义
# 4.1 导入包并设置种子
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# %%
# 4.2 以类的方式定义超参数
class Argparse():
    def __init__(self):
        self.batch_size = 50
        self.epochs = 20
        self.lr = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.input_size = X.shape[1]
        self.hidden1_size = 32
        self.hidden2_size = 64
        self.output_size = y.shape[0]


args = Argparse()


# %%
# 4.3 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(args.input_size, args.hidden1_size),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(args.hidden1_size, args.hidden2_size),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(args.hidden2_size, args.output_size))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        output = self.layer3(x2)
        return output


# %%
# 4.4 定义数据集
class Mydataset(Dataset):
    def __init__(self, flag="train"):
        self.flag = flag
        assert self.flag in ["train", "val"], "not implemented"

        if self.flag == "train":
            self.data = args.X_train
            self.target = args.y_train
        else:
            self.data = args.X_val
            self.target = args.y_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return data, target


# %%
# 4.5 定义训练器
def trainer(modelpath=None):
    train_dataset = Mydataset(flag="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = Mydataset(flag="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = Net().to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_epochs_loss = []
    val_epochs_loss = []
    print("完成前置")

    for epoch in range(args.epochs):
        print("进入循环")
        model.train()
        train_loss, num_samples = 0, 0

        # train
        for X, y in train_loader:
            print("顺利读入训练集", X.shape, y.shape)
            X, y = X.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_samples += y.size(0)

        train_loss /= num_samples
        train_epochs_loss.append(train_loss)
        print(f'第 {epoch + 1}/{args.epochs} 轮，训练损失: {train_loss:.4f}')

        # eval
        model.eval()
        val_loss, num_samples = 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(args.device), y.to(args.device)
                y_pred = model(X)
                loss = criterion(y_pred, y)

                val_loss += loss.item()
                num_samples += y.size(0)

        val_loss /= num_samples
        val_epochs_loss.append(val_loss)
        print(f'第 {epoch + 1}/{args.epochs} 轮，验证损失: {val_loss:.4f}')

        # plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_epochs_loss, "-o", label="train_loss")
        plt.title("train_loss")

        plt.subplot(1, 2, 2)
        plt.plot(train_epochs_loss, "-o", label="train_loss")
        plt.plot(val_epochs_loss, "-o", label="valid_loss")
        plt.title("epoches_loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # save
        if modelpath:
            torch.save(model.state_dict(), modelpath)


# %%
# # 4.5 定义预测器
# def predictor(modelpath=None):
#     model.eval()
# %%
if __name__ == '__main__':
    trainer()