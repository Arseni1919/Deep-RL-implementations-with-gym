from IMPORTS import *
from Reinforece_parameters_class import Net

# ----- INPUT ----- #
num_of_episodes = 50
max_steps = 10000
name_of_env = 'CartPole-v0'


# ----- FUNCTIONS ----- #


def initialise_parameters():
    net = Net()
    return net






def create_sequence(curr_policy, env_name):
    sequence = []
    env = gym.make(env_name)
    env.reset()
    action = env.action_space.sample()
    finish_episode, i = False, 0
    while not finish_episode or i > max_steps:
        # env.render()
        observation, reward, finish_episode, info = env.step(action)
        sequence.append((observation, action, reward))

        action = choose_action(observation, curr_policy)
        print(action, end='')
        i += 1
    print()

        # cv2.imshow('image', observation)
        # cv2.waitKey(0)

    env.close()
    # print(sequence)
    return sequence


def update_policy(net, time_step, sequence_of_episode):  # observation, action, reward = time_step
    # print('time_step', time_step)
    net.zero_grad()
    v = 0
    for i, time_step_tuple in zip(range(len(sequence_of_episode)), sequence_of_episode):
        _, _, reward = time_step_tuple
        pwr = 0
        if i >= time_step:
            v += 0.9**pwr * reward
            pwr += 1
    observation, action, reward = sequence_of_episode[time_step]
    curr_input = net(torch.from_numpy(observation).float()).view(1, -1)
    loss = nn.CrossEntropyLoss()
    soft_max_func = nn.Softmax()
    target = soft_max_func(curr_input)
    target = torch.argmax(target).item()
    target = torch.from_numpy(np.array([target])) #.view(1, -1)

    output = loss(curr_input, target)
    output.backward()

    learning_rate = 3e-4
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate * v * (-1))

    return net


def show_policy_in_action(curr_policy, env_name, n_of_episodes):
    env = gym.make(env_name)

    print(env.action_space)

    for i_episode in range(n_of_episodes):
        print('# ============================ new episode ============================ #')
        g = 0
        env.reset()
        finish_episode = False
        action = env.action_space.sample()
        while not finish_episode:
            env.render()
            time.sleep(0.1)
            observation, reward, finish_episode, info = env.step(action)
            action = choose_action(observation, curr_policy)
            g += reward
        print('sum of rewards in %s is %s' % (i_episode, g))
    env.close()

# ----- ALGORITHM ----- #


def REINFORCE_function(env_name, n_of_episodes):

    policy = initialise_parameters()

    for i_episode in range(n_of_episodes):

        sequence_of_episode = create_sequence(policy, env_name)
        for t in range(len(sequence_of_episode)):
            policy = update_policy(policy, t, sequence_of_episode)

    return policy





if __name__ == '__main__':
    my_policy = REINFORCE_function(name_of_env, num_of_episodes)
    # my_policy = 0
    show_policy_in_action(my_policy, name_of_env, num_of_episodes)







