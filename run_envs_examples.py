import gym
import random
from gym import envs

names = []
for i in envs.registry.all():
    names.append(i._env_name)
random.shuffle(names)

for game in names:
    curr_env = '%s-v0' % game
    print(curr_env)
    try:
        env = gym.make(curr_env)
    except:
        print("[exception]!")
        continue

    for i_episode in range(2):
        observation = env.reset()
        for t in range(100):
            try:
                env.render()
            except:
                continue
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()