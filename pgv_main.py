from IMPORTS import *
from pgv_classes import *

# --------------------- Hyperparameters --------------------- #
learning_rate = 0.01
gamma = 0.99
episodes = 3000
env = gym.make('CartPole-v1')
# env.seed(1)
# torch.manual_seed(1)
running_reward = 10
batch_size = 10

mid_layers = 64

# ----------------------------------------------------------- #

if __name__ == '__main__':

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer

    policy = Policy(env, mid_layers)
    optimizer = optim.Adam(policy.network.parameters(), lr=learning_rate)
    action_space = np.arange(env.action_space.n)
    for episode in range(episodes):
        observation = env.reset()
        env.render()
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = policy.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy.predict(state_tensor))
                    selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    episode + 1, np.mean(total_rewards[-10:])), end="")
    env.close()

    window = 10
    smoothed_rewards = [np.mean(total_rewards[i - window:i + 1]) if i > window
                        else np.mean(total_rewards[:i + 1]) for i in range(len(total_rewards))]

    plt.figure(figsize=(12, 8))
    plt.plot(total_rewards)
    plt.plot(smoothed_rewards)
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
            action_probs = policy.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
            s_0 = s_1

        env.close()

# function REINFORCE
# Initialise θ arbitrarily
# for each episode {s1, a1,r2, ...,sT−1, aT−1,rT } ∼ πθ do
# for t = 1 to T − 1 do
# θ ← θ + α∇θ log πθ(st
# , at)vt
# end for
# end for
# return θ
# end function
