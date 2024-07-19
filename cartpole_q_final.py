import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import pickle
import time
import random

def run(is_training = True, 
        render = False, 
        version = 'v1',
        verbose = True,
        discretizer = 10,
        learning_rate_a = 0.1,
        discount_factor_g = 0.99,
        epsilon_decay_rate = 0.00001,
        iter = None):
    start_time = time.time()

    env = gym.make('CartPole-' + version, render_mode='human' if render else None)

    # Divide position, velocity, pole angle, and pole angular velocity into segments
    pos_space = np.linspace(-2.4, 2.4, discretizer)
    vel_space = np.linspace(-4, 4, discretizer)
    ang_space = np.linspace(-.2095, .2095, discretizer)
    ang_vel_space = np.linspace(-4, 4, discretizer)

    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 11x11x11x11x2 array
    else:
        iter_n = '' if iter is None else f'_{str(iter).zfill(4)}'
        f = open(f'models/q/{version.lower()}/cartpole_{version.lower()}{iter_n}.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = learning_rate_a # alpha or learning rate
    discount_factor_g = discount_factor_g # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = epsilon_decay_rate # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []
    mean_rewards = []

    i = 0

    while True:
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False          # True when reached goal
        rewards = 0

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Choose random action
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action])

            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        rewards_per_episode.append(rewards)
        mean_rewards.append(np.mean(rewards_per_episode[max(0, i-100):i+1]))

        if verbose and is_training and i % 1000 == 0:
            print(f'Episode: {i}  Rewards: {rewards}  Epsilon: {epsilon:0.4f}  Mean Rewards {mean_rewards[-1]:0.1f}')

        threshold_rewards = 475 if version == 'v1' else 195
        if mean_rewards[-1] > threshold_rewards:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon <= 0.00001:
            break 

        i += 1

    env.close()

    # Save Q table to file
    if is_training:
        iter_n = '' if iter is None else f'_{str(iter).zfill(4)}'
        with open(f'models/q/{version.lower()}/cartpole_{version.lower()}{iter_n}.pkl', 'wb') as f:
            pickle.dump(q, f)

    end_time = time.time()
    total_run_time = end_time - start_time

    return i+1, mean_rewards, total_run_time, epsilon

if __name__ == '__main__':
    # execute training stage
    version = 'v0'
    training = True
    plot = True

    params = {'v0': {'discount_factor_g': 0.98, 'discretizer': 10, 'epsilon_decay_rate': 0.0001, 'learning_rate_a': 0.1, 'version': 'v0', 'iter': 529},
              'v1': {'discount_factor_g': 0.98, 'discretizer': 15, 'epsilon_decay_rate': 0.0001, 'learning_rate_a': 0.1, 'version': 'v1', 'iter': 559}}

    episodes, mean_rewards, total_run_time, epsilon = run(is_training = training, 
                                                                render =  False if training else True, 
                                                                version = version,
                                                                verbose = True,
                                                                discretizer = params[version]['discretizer'],
                                                                learning_rate_a = params[version]['learning_rate_a'],
                                                                discount_factor_g = params[version]['discount_factor_g'],
                                                                epsilon_decay_rate = params[version]['epsilon_decay_rate'],
                                                                iter = params[version]['iter'])
    print(f'Episodes: {episodes}')
    print(f'Mean Rewards: {mean_rewards[-1]}')
    print(f'Total Run Time: {total_run_time}')
    print(f'Epsion: {epsilon}')

    if plot and training:
        plt.figure()
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Mean Rewards')
        plt.title(f"CartPole {version} - Iter {params[version]['iter']}")
        plt.savefig(f"plots/q/{version}/cartpole_{version}_{params[version]['iter']}.png")
        plt.close()


