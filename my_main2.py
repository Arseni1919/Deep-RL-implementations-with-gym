from IMPORTS import *
# env = gym.make('procgen:procgen-starpilot-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Tutankham-ram-v0')

env = gym.make('CartPole-v0')
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

plt.plot([1,2,3], [5,6,7])
plt.show()