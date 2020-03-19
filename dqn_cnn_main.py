#!/usr/bin/env python

# DQN for playing OpenAI CartPole. For full writeup, visit:
# https://www.datahubbs.com/deep-q-learning-101/

from dqn_cnn_help import *

# visualise the game with learned policy
# env = gym.make('BreakoutNoFrameskip-v4')
# s_0 = env.reset()
# state_buffer = deque(maxlen=4)
# [state_buffer.append(np.zeros(s_0.size)) for i in range(4)]
# for ep in range(100):
#     s_0 = env.reset()
#     complete = False
#     while not complete:
#         env.render()
#         action = agent.network.get_greedy_action(s_0)
#         s_1, r, complete, _ = env.step(action)
#         s_0 = s_1
#
#     env.close()

def main(argv):
    args = parse_arguments()
    if args.gpu is None or args.gpu == False:
        args.gpu = 'cpu'
    else:
        args.gpu = 'cuda'

    # Initialize environment
    env = gym.make(args.env)

    if args.seed is None:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize DQNetwork
    dqn = QNetwork(env=env,
                   n_hidden_layers=args.hl,
                   n_hidden_nodes=args.hn,
                   learning_rate=args.lr,
                   bias=args.bias,
                   tau=args.tau,
                   device=args.gpu,
                   input_dim=(84, 84))
    # Initialize DQNAgent
    agent = DQNAgent(env, dqn,
                     memory_size=args.memorySize,
                     burn_in=args.burnIn,
                     reward_threshold=args.threshold,
                     path=args.path)
    agent.save_parameters(args)
    print(agent.network)
    # Train agent
    start_time = time.time()

    agent.train(epsilon=args.epsStart,
                gamma=args.gamma,
                # max_episodes=args.maxEps,
                max_episodes=2,
                batch_size=args.batch,
                update_freq=args.updateFreq,
                network_sync_frequency=args.netSyncFreq)
    end_time = time.time()
    # Save results
    agent.save_weights()
    if args.plot:
        agent.plot_rewards()

    x = end_time - start_time
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Peak mean reward: {:.2f}".format(
        max(agent.mean_training_rewards)))
    print("Peak speed: {:.2f}".format(
        max(agent.fps_buffer)))
    print("Training Time: {:02}:{:02}:{:02}\n".format(
        int(hours), int(minutes), int(seconds)))




def parse_arguments():
    parser = ArgumentParser(description='Deep Q Network Argument Parser')
    # Network parameters
    parser.add_argument('--hl', type=int, default=1,
                        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--hn', type=int, default=512,
                        help='An integer number that defines the number of hidden nodes.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--bias', type=str2bool, default=True,
                        help='Boolean to determine whether or not to use biases in network.')
    parser.add_argument('--actFunc', type=str, default='relu',
                        help='Set activation function.')
    parser.add_argument('--gpu', type=str2bool, default=False,
                        help='Boolean to enable GPU computation. Default set to False.')
    parser.add_argument('--seed', type=int,
                        help='Set random seed.')
    # Environment
    parser.add_argument('--env', dest='env', type=str, default='BreakoutNoFrameskip-v4')

    # Training parameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='A value between 0 and 1 to discount future rewards.')
    parser.add_argument('--maxEps', type=int,
                        help='An integer number of episodes to train the agent on.')
    parser.add_argument('--netSyncFreq', type=int, default=2000,
                        help='An integer number that defines steps to update the target network.')
    parser.add_argument('--updateFreq', type=int, default=1,
                        help='Integer value that determines how many steps or episodes' +
                             'must be completed before a backpropogation update is taken.')
    parser.add_argument('--batch', type=int, default=32,
                        help='An integer number that defines the batch size.')
    parser.add_argument('--memorySize', type=int, default=50000,
                        help='An integer number that defines the replay buffer size.')
    parser.add_argument('--burnIn', type=int, default=32,
                        help='Set the number of random burn-in transitions before training.')
    parser.add_argument('--epsStart', type=float, default=0.05,
                        help='Float value for the start of the epsilon decay.')
    parser.add_argument('--epsEnd', type=float, default=0.01,
                        help='Float value for the end of the epsilon decay.')
    parser.add_argument('--epsStrategy', type=str, default='constant',
                        help="Enter 'constant' to set epsilon to a constant value or 'decay'" +
                             "to have the value decay over time. If 'decay', ensure proper" +
                             "start and end values.")
    parser.add_argument('--tau', type=int, default=1,
                        help='Number of states to link together.')
    parser.add_argument('--epsConstant', type=float, default=0.05,
                        help='Float to be used in conjunction with a constant epsilon strategy.')
    parser.add_argument('--window', type=int, default=100,
                        help='Integer value to set the moving average window.')
    parser.add_argument('--plot', type=str2bool, default=True,
                        help='If true, plot training results.')
    parser.add_argument('--path', type=str, default=None,
                        help='Specify path to save results.')
    parser.add_argument('--threshold', type=int, default=195,
                        help='Set target reward threshold for the solved environment.')
    args = parser.parse_args()

    return parser.parse_args()


def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        pass
        # raise ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main(sys.argv)
