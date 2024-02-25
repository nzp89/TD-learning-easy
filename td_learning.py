# TD learning program

import numpy as np
import matplotlib.pyplot as plt

# This is the constant of TD learning.
alpha = 0.5
num_steps = 23
num_episodes = 70

# This is the constant of reward.
stim_time = 6
reward_time = 20

# Weight, stim, and reward are different for each time. (at first, all 0.)
weight = np.zeros(num_steps)
stim = np.zeros(num_steps)
reward = np.zeros(num_steps)

# stim and reward are 1.
stim[stim_time] = 1
reward[reward_time] = 1

# TD_errors and state_value definition.
TD_errors = np.zeros(num_steps)
state_values = np.zeros(num_steps)

# print primary state.
print(f"stim: {stim}")
print(f"reward: {reward}")
print(f"weight: {weight}")

# plot setting (colormap is changeable.)
plt.figure(tight_layout=True)
colormap = plt.get_cmap("cool")

for i in range(num_episodes):
    state_values = np.zeros(num_steps) # you need to reset state_values each episodes.
    for time in range(num_steps - 1):
        for tau in range(time):
            state_values[time] += weight[tau] * stim[time - tau]
    for time in range(num_steps - 1):  
        for tau in range(time):  
            TD_errors[time] = reward[time] + state_values[time + 1] - state_values[time]    
            weight[tau] += alpha * TD_errors[time] * stim[time - tau]
    print(f"trial {i}")
    print(f"state_values: {state_values}")
    print(f"TD_errors: {TD_errors}")
    print(f"weight: {weight}")
    if i % 1 == 0: # each episodes plot the result. (changeable.)
        plt.subplot(211, xlabel="time step", ylabel="TD error")
        plt.plot(TD_errors, color=colormap(i / num_episodes))
        plt.subplot(212, xlabel="time step", ylabel="state value")
        plt.plot(state_values, color=colormap(i / num_episodes))

plt.savefig("td_learning.png", dpi=600, bbox_inches='tight')

