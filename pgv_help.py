from IMPORTS import *


class Policy(nn.Module):
    def __init__(self, env, mid_layers=16):
        super().__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, mid_layers),
            # nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(mid_layers, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

    # def forward(self, x):
    #
    #     model = torch.nn.Sequential(
    #         self.l1,
    #         nn.Dropout(p=0.6),
    #         nn.ReLU(),
    #         self.l2,
    #         # nn.Softmax()
    #         nn.Softmax(dim=-1)
    #     )
    #     return model(x)


def select_action(state, policy):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    # state2 = torch.tensor([1, 0]).type(torch.FloatTensor)
    c = Categorical(state)
    action = c.sample()


    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() > 1:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))

    action = c.sample().item()
    return action


def update_policy(policy, optimizer):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    if rewards.shape[0] > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)

    # Calculate loss
    # logprob = torch.log(
    #     policy.predict(state))
    # selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
    # loss = -selected_logprobs.mean()

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def plot_results(episodes, policy):
    window = int(episodes / 20)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()
    # fig.savefig('results.png')


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()
    # return r