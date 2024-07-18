# Cart pole problem solver - Gymnasium

## Overview
This repository contains solutions to the Cartpole problem from Gymnasium using Q-Learning and
SARSA algorithms. The provided scripts, `cartpole_q.py` and `cartpole_sarsa.py`, implement these
algorithms respectively.

## Q-Learning
Q-Learning is a model-free reinforcement learning algorithm that aims to learn the value of an action
in a particular state. It follows the Bellman equation and updates the Q-values based on the
maximum reward of the next state.

## SARSA
SARSA (State-Action-Reward-State-Action) is another model-free reinforcement learning algorithm.
Unlike Q-Learning, SARSA updates the Q-values based on the action actually taken in the next
state, making it an on-policy method.

## Installation
To execute the provided scripts, ensure you have the required dependencies installed by running:
```bash
pip install -r requirements.txt
```

## Files
- `cartpole_q`: Runs a grid search to find the best parameters for solving the Cartpole problem using the Q-Learning algorithm.
- `cartpole_sarsa`: Runs a grid search to find the best parameters for solving the Cartpole problem using the SARSA algorithm.
- `cartpole_q_final.py`: Implements the Q-Learning algorithm for solving the Cartpole problem with the best parameters found.
- `cartpole_sarsa_final.py`: Implements the SARSA algorithm for solving the Cartpole problem using the best parameters found.

## Usage
After installing the dependencies, run the scripts as follows:
```bash
python cartpole_q_final.py
```
or
```bash
python cartpole_sarsa_final.py
```
You may have to modify the last part of the code, depending on the algorithm and if you want to train or render the result. You need to train first to generate a pickle file and then you can render the cartpole to see the results:
```python
if __name__ == '__main__':
    # run(is_training=True, render=False, version='v0')

    run(is_training=False, render=True, version='v0')
```