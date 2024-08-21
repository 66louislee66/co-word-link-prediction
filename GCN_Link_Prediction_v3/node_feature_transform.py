import pandas as pd
import numpy as np
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df_LDA = pd.read_csv(r'/home/lym/lab/project_work/GCN_Link_Prediction/data/webofsci_train_LDA_features.csv', index_col = 0)
df_LDA = df_LDA.astype(np.float32)
LDA_arr = df_LDA.values.T
LDA_tensor = torch.from_numpy(LDA_arr)
LDA_tensor = LDA_tensor.to(device)

# 初始化自编码器模型
input_dim = LDA_tensor.shape[1]  # 特征维度大小
encoding_dim = 16  # 目标的低维空间维度
model = Autoencoder(input_dim, encoding_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 50  # 训练周期
batch_size = 256  # 批次大小

for epoch in range(num_epochs):
    # 打乱数据
    permutation = torch.randperm(LDA_tensor.size()[0])
    
    for i in range(0, LDA_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_features = LDA_tensor[indices].to(device)
        
        # 正向传播
        outputs = model(batch_features)
        loss = criterion(outputs, batch_features)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 输出训练过程中的损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练好的编码器部分获取压缩的特征
compressed_features = model.encoder(LDA_tensor).detach()
compressed_features = compressed_features.cpu()
compressed_features_numpy = compressed_features.numpy()
df = pd.DataFrame(compressed_features_numpy)
df.to_csv(r'/home/lym/lab/project_work/GCN_Link_Prediction/data/compressed_features.csv', index=False)