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

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 11x11x11x11x2 array
    else:
        iter_n = '' if iter is None else f'_{str(iter).zfill(4)}'
        f = open(f'models/{version.lower()}/cartpole_{version.lower()}{iter_n}.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = learning_rate_a # alpha or learning rate
    discount_factor_g = discount_factor_g # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = epsilon_decay_rate # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []

    i = 0

    # for i in range(episodes):
    while(True):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False          # True when reached goal

        rewards=0

        while(not terminated and rewards < 10000):

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action])

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

            rewards+=reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if verbose and is_training and i%1000==0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.4f}  Mean Rewards {mean_rewards:0.1f}')

        threshold_rewards = 475 if version == 'v1' else 195
        if mean_rewards > threshold_rewards:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon <= 0.00001:
            break 

        i+=1

    env.close()

    # Save Q table to file
    if is_training:
        iter_n = '' if iter is None else f'_{str(iter).zfill(4)}'
        f = open(f'models/{version.lower()}/cartpole_{version.lower()}{iter_n}.pkl', 'wb')
        pickle.dump(q,f)
        f.close()

    # mean_rewards = []
    # for t in range(i):
    #     mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))

    end_time = time.time()

    total_run_time = end_time - start_time
    return i+1, mean_rewards, total_run_time, epsilon


if __name__ == '__main__':
    # run(is_training=True, render=False, version='v0')

    # run(is_training=False, render=True, version='v0')
    param_grid = {'learning_rate_a': [0.0001, 0.001, 0.01, 0.05, 0.1],
                  'discount_factor_g': np.linspace(0.9, 0.99, 10),
                  'epsilon_decay_rate': [0.00001, 0.0001, 0.001, 0.01],
                  'discretizer': [10, 15],
                  'version': ["v0","v1"]}

    param_comb = list(ParameterGrid(param_grid))
    random.seed(150)
    param_comb = random.sample(param_comb, len(param_comb))
    random.seed(None)
    records = pd.DataFrame()
    for iter, param in enumerate(param_comb[:2]):
        print(f"""Version: {param['version']}, 
                Iter: {iter}, 
                learning_rate_a: {param['learning_rate_a']}, 
                discount_factor_g: {param['discount_factor_g']}, 
                epsilon_decay_rate: {param['epsilon_decay_rate']}, 
                discretizer: {param['discretizer']}\n""")
        print('Fit model ...')
        episodes, mean_rewards, total_run_time, epsilon = run(is_training = True, 
                                     render = False, 
                                     version = param['version'],
                                     verbose = True,
                                     discretizer = param['discretizer'],
                                     learning_rate_a = param['learning_rate_a'],
                                     discount_factor_g = param['discount_factor_g'],
                                     epsilon_decay_rate = param['epsilon_decay_rate'],
                                     iter = iter)

        print('Fitting done')

        print('Save record')
        record = pd.DataFrame({'iter': [iter],
                               'version': [param['version']],
                               'params': [str(param)],
                               'episodes': [episodes],
                               'mean_rewards': [mean_rewards],
                               'total_run_time': [total_run_time],
                               'epsilon': [epsilon]})
        
        records = pd.concat([records, record], axis = 0, ignore_index = True)
        
        records.to_csv('output/evaluation.csv', index = False)
        # save 
        #plt.plot(mean_rewards)
        plt.savefig(f"plots/{param['version'].lower()}/cartpole_{param['version'].lower()}{iter}.png")