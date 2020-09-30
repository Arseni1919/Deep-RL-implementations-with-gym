from IMPORTS import *
# env = gym.make('procgen:procgen-starpilot-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Tutankham-ram-v0')

# env = gym.make('CarRacing-v0')
# print(env.action_space)
# #> Discrete(2)
# print(env.observation_space)
# env.reset()
# for _ in range(100):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()
#
# plt.plot([1,2,3], [5,6,7])
# plt.show()
#
# # env = gym.make('procgen:procgen-starpilot-v0')
# # env = gym.make('MountainCarContinuous-v0')
# # env = gym.make('Tutankham-ram-v0')
# # help(gym)
# env = gym.make('procgen:procgen-starpilot-v0')
# GAMMA = 0.9
#
#
# print(env.action_space)
# # print(env.observation_space)
# # print(env.observation_space.high)
# # print(env.observation_space.low)
# for i_episode in range(20):
#     env.reset()
#     for t in range(10000):
#         env.render(mode='human')
#         time.sleep(0.05)
#         # print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         # cv2.imshow('image', observation)
#         # time.sleep(1)
#         # cv2.waitKey(0)
#         if done:
#         #     print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
#
# from gym import spaces
# space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
# x = space.sample()
# assert space.contains(x)
# assert space.n == 8

env = gym.make('MountainCar-v0')
obs = env.reset()
done = False
for i in range(30000):
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    # print(rew)
    if done:
        obs = env.reset()
env.close()