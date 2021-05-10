import gym
import numpy as np

#Hyper parameters
learning_rate = 0.1
discount_f = 0.95
episodes = 30000
n_show = 5000

# Loading env
env = gym.make("MountainCar-v0")
env.reset()

# Dimension of the environment
env_high = env.observation_space.high
env_low = env.observation_space.low
env_goal = env.goal_position


# Discretisation
N_grid = 1000
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


for episode in range(episodes):
    # Training
    discrete_state = discretize(env.reset())
    if episode % n_show == 0:
        print(f"Siamo all'episode {episode}")
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
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = discretize(new_state)
        if not done:
            max_next_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q_value = (1-learning_rate)*current_q + learning_rate*(reward + discount_f*max_next_q)
            q_table[new_discrete_state + (action,)] = new_q_value

        elif new_state[0] >= env_goal:
            q_table[new_discrete_state + (action, )] = 0
            print("Goal reached, MOFOS")
        discrete_state = new_discrete_state
        if to_render:
            env.render()

env.close()
