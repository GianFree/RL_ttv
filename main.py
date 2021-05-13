import gym
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

#Hyper parameters
learning_rate = 0.1
discount_f = 0.95
episodes = 25000
n_show = 2500
epsilon_start = 0.5
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


q_table = np.random.uniform(low=-2, high=0, size=(grid_size + [n_actions]))

epsilon = epsilon_start
ep_reward_list = []
for episode in range(episodes):
    ep_reward = 0.
    # Training
    discrete_state = discretize(env.reset())
    # epsilon = exploration_factor_2(epsilon)
    epsilon = exploration_factor(episode)
    if episode % n_show == 0:
        print(f"Siamo all'episode {episode}")
        print(f"Exploration factor: {exploration_factor(episode)}")
        to_render = True
    else:
        to_render = False
    done = False
    while not done:
        # TODO
        # - come scegliere l'azione
        # - esecuzione dell'azione
        # - calcolo ed update reward
        #print(env.action_space.sample( ))
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
            new_q_value = (1-learning_rate)*current_q + learning_rate*(reward + discount_f*max_next_q)
            q_table[discrete_state + (action,)] = new_q_value

        elif new_state[0] >= env_goal:
            q_table[new_discrete_state + (action, )] = 0
            print(f"Goal reached at episode: {episode}, MOFOS")
        discrete_state = new_discrete_state

    if episode % 5000 == 0:
        np.save(f"q_table_{episode}.npy", q_table)

    ep_reward_list.append(ep_reward)

env.close()


plt.plot(ep_reward_list[::10])

q0 = np.load("q_table_0.npy")
q1 = np.load("q_table_5000.npy")
q2 = np.load("q_table_10000.npy")



sns.heatmap(q0[:,:,2])

sns.heatmap(q1[:,:,2])

sns.heatmap(q2[:,:,2])
