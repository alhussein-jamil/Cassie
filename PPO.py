import torch.nn as nn 
import torch
from torch import optim
import gymnasium as gym 

# Defining the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.logstd)
        return mean, std

# Defining the value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

# Defining some hyperparameters
env_name = "Hopper-v2" # The mujoco environment name
num_workers = 8 # The number of parallel workers
num_steps = 2048 # The number of steps per worker per epoch
num_epochs = 500 # The number of training epochs
gamma = 0.99 # The discount factor
lamda = 0.95 # The GAE parameter
clip_ratio = 0.2 # The PPO clip ratio
pi_lr = 3e-4 # The policy learning rate
vf_lr = 1e-3 # The value function learning rate
train_pi_iters = 80 # The number of policy gradient steps per epoch
train_v_iters = 80 # The number of value function steps per epoch
target_kl = 0.01 # The target KL divergence
device = "cuda" if torch.cuda.is_available() else "cpu" # The device to use

# Creating the environment and getting the observation and action dimensions
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Creating the policy and value networks and their optimizers
pi_net = PolicyNetwork(obs_dim, act_dim).to(device)
vf_net = ValueNetwork(obs_dim).to(device)
pi_optimizer = optim.Adam(pi_net.parameters(), lr=pi_lr)
vf_optimizer = optim.Adam(vf_net.parameters(), lr=vf_lr)

# Defining a function to compute the log probability of an action given a state and a policy network
def compute_log_prob(state, action, net):
    mean, std = net(state)
    dist = torch.distributions.Normal(mean, std)
    log_prob = dist.log_prob(action).sum(axis=-1)
    return log_prob

# Defining a function to compute the advantage estimates using Generalized Advantage Estimation (GAE)
def compute_advantage(rewards, values, dones):
    advantages = torch.zeros_like(rewards).to(device)
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        advantages[t] = delta + gamma * lamda * (1-dones[t]) * last_advantage
        last_advantage = advantages[t]
    return advantages

# Defining a function to compute the discounted returns
def compute_return(rewards, values, dones):
    returns = torch.zeros_like(rewards).to(device)
    last_return = values[-1]
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + gamma * last_return * (1-dones[t])
        last_return = returns[t]
    return returns

# Defining a function to collect trajectories from multiple workers using the current policy network
def collect_trajectories(net):
    states_list, actions_list, rewards_list, dones_list, log_probs_list, values_list = [], [], [], [], [], []
    global_steps_list = []
    # Creating a list of environments for each worker
    envs = [gym.make(env_name) for _ in range(num_workers)]
    # Initializing the states for each worker
    states = torch.tensor([env.reset() for env in envs], dtype=torch.float32).to(device)
    # Looping until we collect enough steps
    global_steps = 0
    while global_steps < num_steps:
        # Getting the actions and log probabilities from the policy network
        with torch.no_grad():
            actions, log_probs = net.get_action(states)
            values = vf_net(states).squeeze(-1)
        # Taking a step in each environment
        next_states, rewards, dones, _ = zip(*[env.step(action.cpu().numpy()) for env, action in zip(envs, actions)])
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        # Storing the trajectories
        states_list.append(states)
        actions_list.append(actions)
        rewards_list.append(rewards)
        dones_list.append(dones)
        log_probs_list.append(log_probs)
        values_list.append(values)
        # Updating the states and the global steps
        states = next_states
        global_steps += num_workers
        global_steps_list.append(global_steps)
    # Getting the last value estimate for each worker
    with torch.no_grad():
        last_values = vf_net(states).squeeze(-1)
    values_list.append(last_values)
    # Concatenating the trajectories
    states = torch.cat(states_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    rewards = torch.cat(rewards_list, dim=0)
    dones = torch.cat(dones_list, dim=0)
    log_probs = torch.cat(log_probs_list, dim=0)
    values = torch.cat(values_list, dim=0)
    # Computing the advantage estimates and the returns
    advantages = compute_advantage(rewards, values, dones)
    returns = compute_return(rewards, values, dones)
    # Normalizing the advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Closing the environments
    for env in envs:
        env.close()
    return states, actions, log_probs, returns, advantages
# Defining a function to update the policy network using the PPO algorithm
def update_policy(states, actions, log_probs_old, returns, advantages):
    # Computing the number of minibatches
    num_minibatches = int(num_steps * num_workers / 64)
    # Looping over the number of policy gradient steps
    for i in range(train_pi_iters):
        # Shuffling the data
        indices = np.random.permutation(num_steps * num_workers)
        states = states[indices]
        actions = actions[indices]
        log_probs_old = log_probs_old[indices]
        returns = returns[indices]
        advantages = advantages[indices]
        # Looping over the minibatches
        for j in range(0, num_steps * num_workers, 64):
            # Getting the minibatch data
            mb_states = states[j:j+64]
            mb_actions = actions[j:j+64]
            mb_log_probs_old = log_probs_old[j:j+64]
            mb_returns = returns[j:j+64]
            mb_advantages = advantages[j:j+64]
            # Computing the log probabilities and the entropy of the current policy
            log_probs, entropy = pi_net.get_log_prob_entropy(mb_states, mb_actions)
            # Computing the ratio of the probabilities
            ratio = torch.exp(log_probs - mb_log_probs_old)
            # Computing the clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * mb_advantages
            surr = -torch.min(surr1, surr2).mean()
            # Taking a gradient step to update the policy network
            pi_optimizer.zero_grad()
            surr.backward()
            pi_optimizer.step()
        # Computing the KL divergence between the old and new policies
        with torch.no_grad():
            kl = (mb_log_probs_old - pi_net.get_log_prob(mb_states, mb_actions)).mean()
        # Checking if the KL divergence is too large
        if kl > 1.5 * target_kl:
            print(f"Early stopping at step {i} due to reaching max kl.")
            break

# Defining a function to update the value network using the PPO algorithm
def update_value(states, returns):
    # Computing the number of minibatches
    num_minibatches = int(num_steps * num_workers / 64)
    # Looping over the number of value function steps
    for i in range(train_v_iters):
        # Shuffling the data
        indices = np.random.permutation(num_steps * num_workers)
        states = states[indices]
        returns = returns[indices]
        # Looping over the minibatches
        for j in range(0, num_steps * num_workers, 64):
            # Getting the minibatch data
            mb_states = states[j:j+64]
            mb_returns = returns[j:j+64]
            # Computing the value estimates and the value loss
            values = vf_net(mb_states).squeeze(-1)
            value_loss = ((values - mb_returns) ** 2).mean()
            # Taking a gradient step to update the value network
            vf_optimizer.zero_grad()
            value_loss.backward()
            vf_optimizer.step()

# Defining a function to evaluate the policy network on a single environment
def evaluate_policy(net):
    env = gym.make(env_name)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            action, _ = net.get_action(state)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        total_reward += reward
    env.close()
    return total_reward

# Training loop
for epoch in range(num_epochs):
    # Collecting trajectories using the current policy network
    states, actions, log_probs_old, returns, advantages = collect_trajectories(pi_net)
    # Updating the policy network using PPO algorithm
    update_policy(states, actions, log_probs_old, returns, advantages)
    # Updating the value network using PPO algorithm
    update_value(states, returns)
    # Evaluating the policy network on a single environment
    eval_reward = evaluate_policy(pi_net)
    print(f"Epoch: {epoch}, Evaluation Reward: {eval_reward}")