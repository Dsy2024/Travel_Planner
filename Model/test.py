import travel_model
import os
import pandas as pd

current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, '..', 'GoogleAPI', 'model_data/taipei/data.csv')
data = pd.read_csv(target_file_path)
data['currentDay'] = pd.NA
data['currentTime'] = pd.NA 

env = travel_model.TravelEnvironment()
agent = travel_model.ActorCriticAgent(env.get_state_size(), len(data))

# 從文件中加載已保存的參數
travel_model.load_agent(agent)
total_rewards = []
left_budgets = []
chosen_data_list = []
excluded_actions = []
num_48 = 0
num_69 = 0
days = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']

for episode in range(1):
    env = travel_model.TravelEnvironment()
    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions(data)
        action = agent.choose_action_test(env.state, valid_actions, excluded_actions)
        next_state, reward, done = env.step(data, action)
        if action is not None:
            state = next_state
            data.loc[action, 'currentDay'] = days[env.current_day]
            data.loc[action, 'currentTime'] = round(env.current_time/60, 1)

    total_rewards.append(round(env.total_reward, 1))
    left_budgets.append(round(env.budget, 1))
    print(env.chosen_actions)
    # print(env.travel_list)
    # if 48 in env.chosen_actions:
    #     num_48 += 1
    # if 69 in env.chosen_actions:
    #     num_69 += 1
    chosen_data = data.iloc[env.chosen_actions]
    chosen_data_list.append(chosen_data)
    
# exit()
# 打印選擇的動作對應的數據
with open('chosen_actions_data.csv', 'w') as f:
    for i, chosen_data in enumerate(chosen_data_list):
        # 寫入 chosen_data 並插入空行
        chosen_data.to_csv(f, header=True if i == 0 else False, index=False)
        f.write('\n')

print(f"Total Reward: {total_rewards}")
print(f"Budget Left: {left_budgets}")
print(f"Average Total Reward: {sum(total_rewards)/len(total_rewards)}")
print(f"Average Budget Left: {sum(left_budgets)/len(left_budgets)}")
# print(f"48 appears: {num_48}")
# print(f"69 appears: {num_69}")