import gymnasium as gym
import numpy as np
import random
import math

# Parámetros del algoritmo
ALPHA = 0.1  # Tasa de aprendizaje
GAMMA = 0.99  # Factor de descuento
EPSILON = 1.0  # Tasa de exploración inicial
EPSILON_DECAY = 0.995  # Decaimiento de epsilon
EPSILON_MIN = 0.01  # Mínimo epsilon
NUM_EPISODES = 1000  # Número de episodios
MAX_STEPS = 200  # Máximo de pasos por episodio
BUCKETS = (1, 1, 6, 12)  # Discretización del espacio de estados

# Crear el entorno
env = gym.make("CartPole-v1")
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# Función para discretizar los estados
def discretize_state(state):
    ratios = [(state[i] + abs(state_bounds[i][0])) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    new_state = [int(round((BUCKETS[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(BUCKETS[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

# Inicializar la Q-table
q_table = np.zeros(BUCKETS + (env.action_space.n,))

# Función para elegir una acción basada en el estado actual
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return env.action_space.sample()  # Explorar
    else:
        return np.argmax(q_table[state])  # Explotar

# Algoritmo Q-Learning
for episode in range(NUM_EPISODES):
    current_state = discretize_state(env.reset()[0])
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = choose_action(current_state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Actualización de la Q-table
        best_q = np.max(q_table[next_state])
        q_table[current_state][action] += ALPHA * (reward + GAMMA * best_q - q_table[current_state][action])

        current_state = next_state
        steps += 1

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    print(f"Episodio: {episode}, Pasos: {steps}, Epsilon: {EPSILON}")

print("Entrenamiento completado")

print('q_table shape', q_table.shape)
# Probar la política aprendida
for _ in range(10):
    state = discretize_state(env.reset()[0])
    done = False
    while not done:
        action = np.argmax(q_table[[state]])
        state, reward, done, _, _ = env.step(action)
        env.render()
env.close()