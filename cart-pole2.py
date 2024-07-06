import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def create_discrete_spaces():
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.2095, 0.2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)
    return pos_space, vel_space, ang_space, ang_vel_space

def initialize_q_table(pos_space, vel_space, ang_space, ang_vel_space, action_space_n, is_training, version):
    if is_training:
        return np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, action_space_n))
    else:
        with open(f'cartpole_{version}.pkl', 'rb') as f:
            return pickle.load(f)

def choose_action(q, state_indices, action_space, epsilon, is_training, rng):
    if is_training and rng.random() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(q[state_indices])

def update_q_table(q, state_indices, action, reward, new_state_indices, learning_rate, discount_factor):
    q[state_indices][action] += learning_rate * (reward + discount_factor * np.max(q[new_state_indices]) - q[state_indices][action])

def run_episode(env, q, pos_space, vel_space, ang_space, ang_vel_space, is_training, rng, epsilon, learning_rate, discount_factor):
    state = env.reset()[0]
    state_indices = tuple(np.digitize(state[i], space) for i, space in enumerate([pos_space, vel_space, ang_space, ang_vel_space]))
    terminated = False
    total_reward = 0

    while not terminated and total_reward < 10000:
        action = choose_action(q, state_indices, env.action_space, epsilon, is_training, rng)
        new_state, reward, terminated, _, _ = env.step(action)
        new_state_indices = tuple(np.digitize(new_state[i], space) for i, space in enumerate([pos_space, vel_space, ang_space, ang_vel_space]))

        if is_training:
            update_q_table(q, state_indices, action, reward, new_state_indices, learning_rate, discount_factor)

        state_indices = new_state_indices
        total_reward += reward

    return total_reward

def save_q_table(q, version):
    with open(f'cartpole_{version}.pkl', 'wb') as f:
        pickle.dump(q, f)

def plot_rewards(rewards_per_episode, version):
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole_{version}.png')

def run(is_training=True, render=False, version='v1'):
    env = gym.make(f'CartPole-{version}', render_mode='human' if render else None)
    pos_space, vel_space, ang_space, ang_vel_space = create_discrete_spaces()
    q = initialize_q_table(pos_space, vel_space, ang_space, ang_vel_space, env.action_space.n, is_training, version)
    
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()
    rewards_per_episode = []
    episode = 0

    while True:
        reward = run_episode(env, q, pos_space, vel_space, ang_space, ang_vel_space, is_training, rng, epsilon, learning_rate, discount_factor)
        rewards_per_episode.append(reward)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if is_training and episode % 100 == 0:
            print(f'Episode: {episode} Rewards: {reward} Epsilon: {epsilon:.2f} Mean Rewards: {mean_rewards:.1f}')

        threshold_rewards = 475 if version == 'v1' else 195
        if mean_rewards > threshold_rewards:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        episode += 1

   
    env.close()

    

    if is_training:
        save_q_table(q, version)

    plot_rewards(rewards_per_episode, version)

if __name__ == '__main__':
    # run(is_training=True, render=False, version='v1')
    run(is_training=False, render=True, version='v1')
