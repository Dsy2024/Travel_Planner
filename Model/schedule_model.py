import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

days = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
labels = ['觀光', '活動', '餐飲']

day_start = 8*60
day_end = 22*60

taipei_location = (25.0330, 121.5654)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# 定義環境
class TravelEnvironment:
    def __init__(self, total_days=3, budget=3000, current_day=0, location=taipei_location, label='觀光'):
        self.total_days = total_days
        self.total_budget = budget
        self.budget = budget
        self.day_left = total_days
        self.current_day = current_day
        self.current_time = day_start
        self.location = location
        self.latitude, self.longitude = location
        try:
            self.label = labels.index(label)
        except ValueError:
            self.label = 0
        self.travel_list = []  # 初始化狀態，可能是空的景點列表
        self.chosen_actions = []
        self.total_reward = 0
        self.bonus_reward = total_days
        self.state = (
            normalize(self.budget, 0, 10000),  # 假設 budget 的範圍是 0 到 10000
            normalize(self.day_left, 0, 10),  # 假設 day_left 的範圍是 0 到 10
            normalize(self.current_day, 0, 6),  # 假設 current_day 是一周中的 0 到 6
            normalize(self.current_time, 0, 1440),  # 假設一天中的分鐘數（0 到 1440）
            normalize(self.latitude, -90, 90),  # 經緯度的合理範圍
            normalize(self.longitude, -180, 180),
            float(self.label)  # 類別索引轉換為浮點數（0, 1, 2 不需要特別的歸一化）
        )   

    def reset(self):
        self.budget = self.total_budget
        self.day_left = self.total_days
        self.current_time = day_start
        self.latitude, self.longitude = self.location
        self.travel_list = []
        self.chosen_actions = []
        self.total_reward = 0
        self.bonus_reward = self.total_days
        self.state = (
            normalize(self.budget, 0, 10000),  # 假設 budget 的範圍是 0 到 10000
            normalize(self.day_left, 0, 10),  # 假設 day_left 的範圍是 0 到 10
            normalize(self.current_day, 0, 6),  # 假設 current_day 是一周中的 0 到 6
            normalize(self.current_time, 0, 1440),  # 假設一天中的分鐘數（0 到 1440）
            normalize(self.latitude, -90, 90),  # 經緯度的合理範圍
            normalize(self.longitude, -180, 180),
            float(self.label)  # 類別索引轉換為浮點數（0, 1, 2 不需要特別的歸一化）
        )
        return self.state

    def get_state_size(self):
        return len(self.state)
    
    def step(self, data, action):
        if action == None:
            self.current_time += 60
            reward = 0
        else:
            attraction = data.loc[action]
            visit_time = attraction['recommend'] * 60  # 轉換為分鐘
            
            # 移除已選擇的景點
            # data.drop(action, inplace=True)
            
            self.budget -= attraction['price']
            self.current_time += visit_time
            self.latitude, self.longitude = (attraction['latitude'], attraction['longitude'])
            self.travel_list.append(attraction['basicName'])
            self.chosen_actions.append(action)
            reward = self.calculate_reward(attraction)
            self.total_reward += reward
        done = False
        if self.current_time >= day_end:
            self.day_left -= 1
            if self.day_left > 0 and len(data) > 0:
                self.current_day = (self.current_day + 1) % 7
                self.current_time = day_start
            else:
                done = True  # 沒有足夠時間，遊戲結束    
        
        self.state = (
            normalize(self.budget, 0, 10000),  # 假設 budget 的範圍是 0 到 10000
            normalize(self.day_left, 0, 10),  # 假設 day_left 的範圍是 0 到 10
            normalize(self.current_day, 0, 6),  # 假設 current_day 是一周中的 0 到 6
            normalize(self.current_time, 0, 1440),  # 假設一天中的分鐘數（0 到 1440）
            normalize(self.latitude, -90, 90),  # 經緯度的合理範圍
            normalize(self.longitude, -180, 180),
            float(self.label)  # 類別索引轉換為浮點數（0, 1, 2 不需要特別的歸一化）
        )
        return self.state, reward, done

    def calculate_reward(self, attraction):
        if self.budget/self.total_budget < 0.1:
            return 0
        reward = attraction['rating']  # 基本獎勵是評分
        if attraction['label'] == labels[self.label]:  # 假設符合類型的景點獲得bonus
            reward += self.bonus_reward
            if self.bonus_reward > 0:
                self.bonus_reward -= 0.5
        reward *= (1 + 5*attraction['price']/self.total_budget)
        return reward

    def get_valid_actions(self, data):
        valid_actions = []
        for action in data.index:
            if action in self.chosen_actions:
                continue  # 跳過已選擇的動作
            
            open_time = data.loc[action][f'{days[self.current_day]}Open']
            stay_time = data.loc[action]['recommend']*60 + self.current_time
            close_time = data.loc[action][f'{days[self.current_day]}Close']
            price = data.loc[action]['price']
            
            if open_time <= self.current_time and stay_time <= close_time and self.budget > price:
                valid_actions.append(action)
        return valid_actions


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_out = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_out(x), dim=-1)
        return action_probs

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_out = nn.Linear(128, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_out(x)
        return value

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=0.0003, lr_critic=0.001):
        self.gamma = 0.95                # 折扣因子，用于计算未来奖励的当前价值
        self.eps_clip = 0.2          # PPO的剪裁范围，这是PPO特有的一个重要参数，用于限制策略更新的幅度
        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.policy_losses = []
        self.value_losses = []

    def choose_action(self, state):
        # 将状态转换为 tensor 并添加维度 (batch_size, state_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 通过策略网络获取动作概率
        action_probs = self.actor(state)
        
        # 创建分布并根据概率采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # 返回动作和对应的 log 概率
        return action.item(), dist.log_prob(action).item()

    def update(self, transitions):
        states, actions, old_log_probs, returns, advantages = transitions

        # Convert lists to tensor
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # Get new log probabilities and state values
        new_probs = self.policy(states)
        new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        state_values = self.critic(states).squeeze(-1)

        # Calculate the ratio
        ratios = torch.exp(new_log_probs - old_log_probs.detach())

        # Calculate surrogate loss
        surr = torch.clamp(ratios, 1-self.clip_param, 1+self.clip_param) * advantages
        policy_loss = -surr.mean()

        # Calculate value loss
        value_loss = (returns - state_values).pow(2).mean()

        # 更新
        self.optimizer_actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

        return policy_loss.item(), value_loss.item()
