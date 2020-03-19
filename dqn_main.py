from IMPORTS import *
from dqn_help import *


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    buffer = experienceReplayBuffer(memory_size=10000, burn_in=1000)
    dqn = Q_func(env, learning_rate=1e-3)
    agent = DQNAgent(env, dqn, buffer, reward_threshold=195)
    training_rewards, mean_training_rewards = agent.train(max_episodes=5000, network_update_frequency=1,
                network_sync_frequency=2500)

    plt.figure(figsize=(12, 8))
    plt.plot(training_rewards)
    plt.plot(mean_training_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show()

    # visualise the game with learned policy
    env = gym.make('CartPole-v1')
    for ep in range(100):
        s_0 = env.reset()
        complete = False
        while not complete:
            env.render()
            action = agent.network.get_greedy_action(s_0)
            s_1, r, complete, _ = env.step(action)
            s_0 = s_1

        env.close()
