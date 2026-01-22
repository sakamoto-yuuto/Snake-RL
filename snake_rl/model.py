import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 1. 第一层连线：把 11 个状态 变成 256 个特征
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        # 2. 第二层连线：把 256 个特征 变成 3 个动作的打分
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 就是输入的状态 (State)
        
        # 第一步：过第一层网络
        x = self.linear1(x)
        
        # 第二步：过激活函数 (ReLU)
        # 你可以把 ReLU 想象成一个过滤器，把负数变成0，正数保留。
        # 这一步是为了让网络学会非线性的复杂逻辑。
        x = F.relu(x)
        
        # 第三步：过第二层网络，直接输出三个动作的分数
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # 这个函数用来把训练好的大脑存到硬盘上，防止关机就忘了
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


import torch.optim as optim

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr        # 学习率 (Learning Rate)：学得太快容易甚至走火入魔，太慢学不会
        self.gamma = gamma  # 折扣率：多大程度上重视未来的奖励
        self.model = model
        
        # Adam 是一个优化器，它就像一个聪明的老师，通过调整参数来减少错误
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # 损失函数 (MSE)：用来计算“预测值”和“真实值”的差距
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 把数据转换成 PyTorch 能看懂的 Tensor (张量) 格式
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # 如果只传入了一步的数据，就把它扩充成 "一批" 数据 (Batch)
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. 用现在的状态预测 Q 值
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 核心公式：Q_new = 奖励 + gamma * 未来那一步的最大预期奖励
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # 更新我们要逼近的目标值
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2. 反向传播 (Backpropagation)
        # 清空之前的梯度 -> 计算差距(loss) -> 算出如何调整参数 -> 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()