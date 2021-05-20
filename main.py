import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import seaborn as sns

#Hyper parameters
learning_rate = 0.1
discount_f = 0.95
episodes = 20000
n_show = 2500
epsilon_start = 0.1 # fixed for investigating hyperparameters (old = 0.5)
epsilon_end = 0.1
start_episode = 0
end_episode = episodes//2


def exploration_factor(episode):
    exp_fac = epsilon_start + episode*(epsilon_end - epsilon_start)/(end_episode-start_episode)
    if exp_fac > 0:
        return exp_fac
    else:
        return epsilon_end

def exploration_factor_2(epsilon):
    return epsilon*.9



# Loading env
env = gym.make("MountainCar-v0")
env.reset()

# Dimension of the environment
env_high = env.observation_space.high
env_low = env.observation_space.low
env_goal = env.goal_position

print(env_goal)
# Discretisation
N_grid = 20
grid_size = [N_grid]*len(env_high)
n_actions = env.action_space.n
discrete_size  = (env_high - env_low)/grid_size

def discretize(state):
    """
    Input: state ("continous")
    Output: discrete_state ( in the grid )
    """
    discrete_state = (state - env_low)/discrete_size
    return tuple(discrete_state.astype(np.int))

learning_rates = [0.1]
discount_factors = [0.95]

episode_dict = dict()

for l_rate, d_factor in product(learning_rates, discount_factors):
    print(f"Doing learning rate:{l_rate} with discount factor: {d_factor}")

    q_table = np.random.uniform(low=-2, high=0, size=(grid_size + [n_actions]))

    epsilon = epsilon_start
    ep_reward_list = []
    for episode in range(episodes):
        ep_reward = 0.
        # Training
        discrete_state = discretize(env.reset())
        # epsilon = exploration_factor_2(epsilon)
        # epsilon = exploration_factor(episode)
        if episode % n_show == 0:
            print(f"Siamo all'episode {episode}")
            #print(f"Exploration factor: {exploration_factor(episode)}")
            to_render = True
        else:
            to_render = False
        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0,n_actions)

            # print(action)
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = discretize(new_state)

            if to_render:
                env.render()

            ep_reward += reward
            if not done:
                max_next_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q_value = (1-l_rate)*current_q + l_rate*(reward + d_factor*max_next_q)
                q_table[discrete_state + (action,)] = new_q_value

            elif new_state[0] >= env_goal:
                q_table[new_discrete_state + (action, )] = 0
                print(f"Goal reached at episode: {episode}, MOFOS")
            discrete_state = new_discrete_state

        if episode == (episodes - 1):
            np.save(f"q_table_{l_rate}_{d_factor}.npy", q_table)

        ep_reward_list.append(ep_reward)

    #after investigating the pair l_rate and d_factor, we save the result
    episode_dict[f"{l_rate}-{d_factor}"] = ep_reward_list

env.close()

#df = pd.DataFrame.from_dict(episode_dict)
#df.to_csv("q-learning_combination.csv")
df = pd.read_csv("q-learning_combination.csv", index_col=0)
df.head()



df.rolling(200).mean().plot(xlabel='Episodes',ylabel='Reward',figsize = (15,10))


df.filter(regex="0.0").rolling(200).mean().plot(xlabel='Episodes',ylabel='Reward',figsize = (15,10))

df.filter(regex="0.9$").rolling(200).mean().plot(xlabel='Episodes',ylabel='Reward',figsize = (15,10))

df.filter(regex="^0.9").rolling(1).mean().plot(xlabel='Episodes',ylabel='Reward',figsize = (15,10))


##################
#
#    SARSA
#
######################
#Hyper parameters
learning_rate = 0.1
discount_f = 0.95
episodes = 20000
n_show = 2500
epsilon_start = 0.1 # fixed for investigating hyperparameters (old = 0.5)
epsilon_end = 0.1
start_episode = 0
end_episode = episodes//2


def exploration_factor(episode):
    exp_fac = epsilon_start + episode*(epsilon_end - epsilon_start)/(end_episode-start_episode)
    if exp_fac > 0:
        return exp_fac
    else:
        return epsilon_end

def exploration_factor_2(epsilon):
    return epsilon*.9



# Loading env
env = gym.make("MountainCar-v0")
env.reset()

# Dimension of the environment
env_high = env.observation_space.high
env_low = env.observation_space.low
env_goal = env.goal_position

print(env_goal)
# Discretisation
N_grid = 20
grid_size = [N_grid]*len(env_high)
n_actions = env.action_space.n
discrete_size  = (env_high - env_low)/grid_size

def discretize(state):
    """
    Input: state ("continous")
    Output: discrete_state ( in the grid )
    """
    discrete_state = (state - env_low)/discrete_size
    return tuple(discrete_state.astype(np.int))

l_rate = 0.1
d_factor = 0.95

episode_dict = dict()

print(f"Doing learning rate:{l_rate} with discount factor: {d_factor}")

q_table = np.random.uniform(low=-2, high=0, size=(grid_size + [n_actions]))

epsilon = epsilon_start
ep_reward_list = []

#TEST action
for episode in range(episodes):
    ep_reward = 0.
    # Training
    discrete_state = discretize(env.reset())
    # epsilon = exploration_factor_2(epsilon)
    # epsilon = exploration_factor(episode)
    if episode % n_show == 0:
        print(f"Siamo all'episode {episode}")
        #print(f"Exploration factor: {exploration_factor(episode)}")
        to_render = True
    else:
        to_render = False
    done = False


    while not done:

        if np.random.random() < epsilon:
            action = np.random.randint(0,n_actions)
        else:
            action = np.argmax(q_table[discrete_state]) #1

        # print(action)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = discretize(new_state)

        if np.random.random() < epsilon:
            new_action = np.random.randint(0,n_actions)
        else:
            new_action = np.argmax(q_table[new_discrete_state]) #1

        if to_render:
            env.render()

        ep_reward += reward
        if not done:
            next_q = q_table[new_discrete_state+(new_action,)]
            current_q = q_table[discrete_state + (action,)]
            new_q_value = (1-l_rate)*current_q + l_rate*(reward + d_factor*next_q)
            q_table[discrete_state + (action,)] = new_q_value

        elif new_state[0] >= env_goal:
            q_table[new_discrete_state + (action, )] = 0
            print(f"Goal reached at episode: {episode}, MOFOS")

        discrete_state = new_discrete_state
        #action = new_action

    if episode == (episodes - 1):
        np.save(f"q_table-SARSA_{l_rate}_{d_factor}.npy", q_table)

    ep_reward_list.append(ep_reward)

    #after investigating the pair l_rate and d_factor, we save the result
    #episode_dict[f"SARSA{l_rate}-{d_factor}"] = ep_reward_list

env.close()
