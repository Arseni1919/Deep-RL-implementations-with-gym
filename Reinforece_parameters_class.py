from IMPORTS import *

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 4)  # 6*6 from image dimension
        self.fc2 = nn.Linear(4, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def choose_action(observation, net):
    # print('observation:', observation)
    out = net(torch.from_numpy(observation).float())

    # soft_max_func = nn.Softmax()
    # target = soft_max_func(out)
    action = np.random.choice([0, 1], p=np.squeeze(out.detach().numpy()))
    # print(action)
    # print('out: ', torch.argmax(target))
    return action
    # return torch.argmax(target).item()


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # probs = self.forward(Variable(state))
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob