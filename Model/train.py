import travel_model
import os
import chardet
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, '..', 'GoogleAPI', 'model_data/taipei/data.csv')
data = pd.read_csv(target_file_path)
num_episodes = 1000
batch_size = 32

# 訓練模型
env = travel_model.TravelEnvironment()
agent = travel_model.ActorCriticAgent(env.get_state_size(), len(data))

total_rewards = []
left_budgets = []

for episode in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    transitions = []
    
    while not done:
        # print(f"Current State: {state}")
        valid_actions = env.get_valid_actions(data)
        # print(f"Valid actions: {valid_actions}")
        action = agent.choose_action(env.state, valid_actions)
        # print(f"Action Taken: {action}")
        next_state, reward, done = env.step(data, action)
        if action is not None:
            transitions.append((state, action, reward, done, next_state))
            state = next_state
        # print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")

        if done or len(transitions) >= batch_size:
            agent.update(transitions)
            transitions = []

    total_rewards.append(env.total_reward)
    left_budgets.append(env.budget)

# 畫出回報隨著訓練進展的變化
episodes=range(num_episodes)
# 創建一個畫布，並設置兩個子圖
plt.figure(figsize=(12, 8))

# 子圖 1: 總回報
plt.subplot(2, 1, 1)  # (行, 列, 當前子圖編號)
plt.plot(episodes, total_rewards, label='Total Reward', color='b')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress - Total Reward')
plt.grid(True)

# 子圖 2: 剩餘預算
plt.subplot(2, 1, 2)
plt.plot(episodes, left_budgets, label='Budget Left', color='r')
plt.xlabel('Episode')
plt.ylabel('Budget Left')
plt.title('Training Progress - Budget Left')
plt.grid(True)

# 調整布局，避免重疊
plt.tight_layout()

# 保存圖形
plt.savefig('training_progress.png')
plt.close()

def plot_loss(loss, title, filename):
    plt.plot(loss, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_loss(agent.policy_losses, 'Policy Loss Over Time', 'policy_loss.png')
plot_loss(agent.value_losses, 'Value Loss Over Time', 'value_loss.png')

travel_model.save_agent(agent)