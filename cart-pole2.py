import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Definir la red neuronal para DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Parámetros comunes
env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
n_episodes = 1000
max_steps = 200
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Q-learning con discretización
num_bins = 10
alpha = 0.1
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -0.5]
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], 0.5]

def create_bins(num_bins, lower_bounds, upper_bounds):
    bins = []
    for i in range(len(lower_bounds)):
        bins.append(np.linspace(lower_bounds[i], upper_bounds[i], num_bins))
    return bins

bins = create_bins(num_bins, lower_bounds, upper_bounds)
q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, action_size))

def discretize_state(state, bins):
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discretized)

def choose_action_qlearning(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    discretized_state = discretize_state(state, bins)
    return np.argmax(q_table[discretized_state])

# Agente DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.model.apply(self.init_weights)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy()[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state).detach()).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            target_f[0][action] = target
            target_f = torch.FloatTensor(target_f)
            self.model.train()
            output = self.model(torch.FloatTensor(state).unsqueeze(0))
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent(state_size, action_size)

# Entrenamiento de Q-learning y DQN
rewards_qlearning = []
rewards_dqn = []

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward_qlearning = 0
    total_reward_dqn = 0

    # Q-learning
    for step in range(max_steps):
        action = choose_action_qlearning(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward_qlearning += reward
        discretized_state = discretize_state(state, bins)
        discretized_next_state = discretize_state(next_state, bins)
        best_next_action = np.argmax(q_table[discretized_next_state])
        q_table[discretized_state][action] = q_table[discretized_state][action] + alpha * (reward + gamma * q_table[discretized_next_state][best_next_action] - q_table[discretized_state][action])
        state = next_state
        if done:
            break
    rewards_qlearning.append(total_reward_qlearning)

    # DQN
    state, _ = env.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward_dqn += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory) > 64:
            agent.replay(64)
    rewards_dqn.append(total_reward_dqn)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Q-learning Reward: {total_reward_qlearning}, DQN Reward: {total_reward_dqn}, Epsilon: {epsilon}")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Visualización de resultados
plt.plot(rewards_qlearning, label='Q-learning')
plt.plot(rewards_dqn, label='DQN')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('Reinforcement Learning Performance Comparison')
plt.show()
