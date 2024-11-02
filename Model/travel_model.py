import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

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
        self.actor = nn.Linear(128, action_size)  # Actor 預測策略

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        # 減去最大值提高數值穩定性
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        policy = torch.softmax(logits, dim=-1)  # 策略輸出概率分佈
        return policy

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.critic = nn.Linear(128, 1)  # Critic 預測值函數

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.critic(x)  # 值函數輸出
        return value

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_model = ActorNetwork(state_size, action_size)
        self.critic_model = CriticNetwork(state_size)
        self.actor_optim = optim.Adam(self.actor_model.parameters(), lr=0.001)
        self.critic_optim = optim.Adam(self.critic_model.parameters(), lr=0.001)
        self.gamma = 0.99  # 折扣因子
        self.policy_losses = []
        self.value_losses = []

    def choose_action(self, state, valid_actions):
        state = torch.FloatTensor(state).unsqueeze(0)
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            print(f"Invalid values detected in state: {state}")
        policy = self.actor_model(state)
        # print(policy)
    
        # 檢查 policy 是否有無效值
        if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
            print(f"Invalid values detected in policy: {policy}")
            exit()
            policy = torch.softmax(torch.zeros_like(policy), dim=-1)  # 重置為均勻分佈

        # 將策略的無效動作和已選動作的概率設置為零
        mask = torch.zeros(self.action_size)
        mask[valid_actions] = 1  # 只允許有效的動作
        valid_policy = policy * mask  
        # print(valid_policy)

        # 檢查是否有可選動作
        if valid_policy.sum().item() == 0:
            print("Warning: No valid actions available.")
            return None
        
        # 重新歸一化有效動作的概率分佈
        valid_policy /= valid_policy.sum()

        # 再次檢查 valid_policy 是否有無效值
        if torch.any(torch.isnan(valid_policy)) or torch.any(torch.isinf(valid_policy)) or torch.any(valid_policy < 0):
            print(f"Invalid probabilities in valid_policy: {valid_policy}")
            return random.choice(valid_actions)

        # 根據有效動作的概率分佈隨機選擇一個動作
        action = torch.multinomial(valid_policy, 1).item()
        return action
    
    def choose_action_test(self, state, valid_actions, excluded_actions=None):
        # 使用 torch.no_grad() 來禁用梯度計算
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
                print(f"Invalid values detected in state: {state}")
            policy = self.actor_model(state)
            # print(policy)
    
        # 檢查 policy 是否有無效值
        if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
            print(f"Invalid values detected in policy: {policy}")
            exit()

        # 將策略的無效動作和已選動作的概率設置為零
        mask = torch.zeros(self.action_size)
        mask[valid_actions] = 1  # 只允許有效的動作
        mask[excluded_actions] *= 0.1 # 降低已排除景點的選擇概率
        valid_policy = policy * mask  
        # print(valid_policy)

        # 檢查是否有可選動作
        if valid_policy.sum().item() == 0:
            print("Warning: No valid actions available.")
            return None
        
        # 重新歸一化有效動作的概率分佈
        valid_policy /= valid_policy.sum()

        # 再次檢查 valid_policy 是否有無效值
        if torch.any(torch.isnan(valid_policy)) or torch.any(torch.isinf(valid_policy)) or torch.any(valid_policy < 0):
            print(f"Invalid probabilities in valid_policy: {valid_policy}")
            return random.choice(valid_actions)

        # 根據有效動作的概率分佈隨機選擇一個動作
        action = torch.multinomial(valid_policy, 1).item()
        return action

    def compute_loss(self, states, actions, rewards, dones, next_states):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        next_states = torch.FloatTensor(next_states)

        # 計算當前狀態的價值
        values = self.critic_model(states)
        if torch.any(torch.isnan(values)) or torch.any(torch.isinf(values)):
            print("Invalid values detected in values:", values)
        values = values.squeeze()

        # 計算下一狀態的值
        next_values = self.critic_model(next_states)
        next_values = next_values.squeeze()

        # 計算 TD 目標
        target_values = rewards + self.gamma * next_values * (1 - dones)
        if torch.any(torch.isnan(target_values)) or torch.any(torch.isinf(target_values)):
            print("Invalid values detected in target_values:", target_values)
        target_values = target_values.detach()

        # 計算價值損失
        value_loss = (values - target_values).pow(2).mean()

        # 計算策略損失
        policies = self.actor_model(states)
        log_policies = torch.log(policies.gather(1, actions.unsqueeze(1)) + 1e-8)
        advantage = (target_values - values).detach()
        if torch.any(torch.isnan(advantage)) or torch.any(torch.isinf(advantage)):
            print("Invalid values detected in advantage:", advantage)
        policy_loss = -(log_policies.squeeze() * advantage).mean()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        return policy_loss, value_loss

    def update(self, transitions):
        # 提取每個批次中的元素
        states, actions, rewards, dones, next_states = zip(*transitions)

        # 計算損失
        policy_loss, value_loss = self.compute_loss(states, actions, rewards, dones, next_states)

        # 反向傳播和更新
        self.critic_optim.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            policy_loss.backward()
        self.actor_optim.step()

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        # 打印每一層的梯度大小
        for name, param in self.actor_model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm()}")
        for name, param in self.critic_model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm()}")

def save_agent(agent, filename="agent.pth"):
    # 保存模型狀態字典
    torch.save({
        'actor_model_state_dict': agent.actor_model.state_dict(),
        'critic_model_state_dict': agent.critic_model.state_dict(),
    }, filename)
    print(f"Agent has been saved to {filename}")

def load_agent(agent, filename="agent.pth"):
    # 加載模型狀態字典
    checkpoint = torch.load(filename, weights_only=False)
    # 恢復Actor和Critic模型的狀態字典
    agent.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
    agent.critic_model.load_state_dict(checkpoint['critic_model_state_dict'])
    # 將模型設置為評估模式
    agent.actor_model.eval()
    agent.critic_model.eval()
    print(f"Agent has been loaded from {filename}")
