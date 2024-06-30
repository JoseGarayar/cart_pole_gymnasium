import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Discretización del espacio de estados
def discretize_state(state, bins):
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discretized)

# Crear bins para discretización
def create_bins(num_bins, lower_bounds, upper_bounds):
    bins = []
    for i in range(len(lower_bounds)):
        bins.append(np.linspace(lower_bounds[i], upper_bounds[i], num_bins))
    return bins

# Definir el entorno y parámetros
env = gym.make('CartPole-v1', render_mode='human')
n_episodes = 1000
max_steps = 200
gamma = 0.99
alpha = 0.1
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_bins = 10

# Definir los límites para los bins
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -0.5]
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], 0.5]

bins = create_bins(num_bins, lower_bounds, upper_bounds)
q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

# Función para elegir una acción
def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    discretized_state = discretize_state(state, bins)
    return np.argmax(q_table[discretized_state])

# Entrenamiento del agente Q-learning
rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        discretized_state = discretize_state(state, bins)
        discretized_next_state = discretize_state(next_state, bins)
        best_next_action = np.argmax(q_table[discretized_next_state])
        q_table[discretized_state][action] = q_table[discretized_state][action] + alpha * (reward + gamma * q_table[discretized_next_state][best_next_action] - q_table[discretized_state][action])
        state = next_state
        if done:
            break
    rewards.append(total_reward)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")

# Visualización de la solución
for _ in range(5):
    state, _ = env.reset()
    for time in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        if done:
            break
env.close()

# Graficar el rendimiento
plt.plot(rewards)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()  # Mostrar el gráfico directamente
