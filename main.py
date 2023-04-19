import copy
import gymnasium as gym
import itertools
import numpy as np
import random
import time
import torch

from gymnasium.wrappers import TransformObservation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

lr = 5e-4
weight_decay=lr / 10.0

trajectory_chunk_size = 1024

batch_size = 64
no_epochs = 8
clip_epsilon = 0.2
gamma = 0.99 # how far into the future the network cares about
lmbda = 0.95 # used in advantage estimate

C_VF = 0.01
C_EN = 0.1

checkpoint_time = 10000

ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME)

print(env.observation_space.shape)
print(env.action_space)

class TransitionMemory:
    def __init__(self):
        self.clear()

    def __len__(self):
        return len(self.state_memory)

    def append_transition(self, s, a, r, s_new):
        self.state_memory.append(s)
        self.action_memory.append(a)
        self.reward_memory.append(r)
        self.state_new_memory.append(s_new)

    def clear(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.state_new_memory = []

    def get_transition(self, index):
        return (self.state_memory[index], self.action_memory[index], self.reward_memory[index], self.state_new_memory[index])

    def get_transition_list(self):
        return [self.state_memory, self.action_memory, self.reward_memory, self.state_new_memory]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.policy(x)

class ValueFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.model(x)

class Agent:
    def __init__(self):
        self.policy = Policy().to(device)
        self.value_function = ValueFunction().to(device)

        self.vf_loss_fn = torch.nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value_function.parameters()), lr=lr, weight_decay=weight_decay, maximize=True)

        self.transition_memory = []

    def get_action_probabilities(self, s):
        s = torch.as_tensor(s).to(device)
        return self.policy(s)

    def get_action(self, s):
        action_probs = self.get_action_probabilities(s).cpu().detach().numpy()
        return np.random.choice(len(action_probs), p=action_probs)

    def get_action_top(self, s):
        action_probs = self.get_action_probabilities(s).cpu().detach().numpy()
        return np.argmax(action_probs)

    def get_value(self, s, squeeze=False):
        s = torch.as_tensor(s).to(device)
        result = self.value_function(s)

        if squeeze:
            return torch.squeeze(result)
        else:
            return result

    def get_advantage_estimates(self, transition_memory):
        dts = []
        for i in range(len(transition_memory)):
            s_t, a_t, r_t, s_t1 = transition_memory.get_transition(i)
            v_t, v_t1 = self.get_value(s_t, squeeze=True), self.get_value(s_t1, squeeze=True)
            dts.append(-v_t + r_t + gamma * v_t1)

        advantage_estimates = [None] * trajectory_chunk_size
        advantage_estimates[-1] = dts[-1]
        for t in range(len(dts) - 2, -1, -1):
            advantage_estimates[t] = (dts[t] + (gamma * lmbda) * advantage_estimates[t + 1]).detach() # remove gradients since the policy loss will interfere with the value function otherwise

        return torch.as_tensor(advantage_estimates)

    def update(self, transition_memory):
        old_policy = copy.deepcopy(self.policy)

        ads = self.get_advantage_estimates(transition_memory)
        transition_sequence_ads = list(map(lambda x: torch.as_tensor(np.array(x)).to(device), transition_memory.get_transition_list()))
        transition_sequence_ads.append(ads)

        for epoch in range(no_epochs):
            indices = torch.randperm(transition_sequence_ads[0].size(dim=0)) # generate random order for training examples

            for batch_indices in indices.view(batch_size, -1):
                s = transition_sequence_ads[0][batch_indices]
                a = torch.unsqueeze(transition_sequence_ads[1][batch_indices], dim=0)
                r = transition_sequence_ads[2][batch_indices]
                s_new = transition_sequence_ads[3][batch_indices]
                ad = transition_sequence_ads[4][batch_indices]

                probs_old = old_policy(s)
                probs = self.policy(s)

                log_probs = torch.log2(probs)
                entropy = torch.sum(-probs * log_probs, 1)

                ratio = torch.gather(probs, 1, a) / torch.gather(probs_old, 1, a)
                ratio = torch.squeeze(ratio, 0)

                clipped_ratio = (
                        (1.0 + torch.sign(ad)) * torch.minimum(ratio, torch.full_like(ratio, 1.0 + clip_epsilon)) + # if advantage is positive
                        (1.0 - torch.sign(ad)) * torch.maximum(ratio, torch.full_like(ratio, 1.0 - clip_epsilon))   # if advantage is negative
                    ) / 2.0

                loss_clip = torch.minimum(ad * ratio, ad * clipped_ratio)

                v_t = torch.squeeze(self.value_function(s))
                v_t1 = torch.squeeze(self.value_function(s_new))
                target = (r + gamma * v_t1).detach()
                diff = target - v_t
                loss_vf = diff * diff

                loss = torch.mean(loss_clip - C_VF * loss_vf + C_EN * entropy)
                
                writer.add_scalar("Policy Entropy", torch.mean(entropy), t)
                writer.add_scalar("Value Loss", torch.mean(loss_vf), t)
                writer.add_scalar("Policy Loss", torch.mean(loss_clip), t)
                writer.add_scalar("Total Loss", loss, t)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def test_agent(agent, n, show=False, use_argmax=False):
    if show:
        test_env = gym.make(ENV_NAME, render_mode="human")
    else:
        test_env = gym.make(ENV_NAME)

    returns = []
    for _ in tqdm(range(n)):
        state, info = test_env.reset()
        
        r = 0.0
        while True:
            if show:
                print(state)

            if use_argmax:
                action = agent.get_action_top(state)
            else:
                action = agent.get_action(state)

            state, reward, terminated, truncated, _ = test_env.step(action)
            r += reward

            if terminated or truncated:
                break

        returns.append(r)

    test_env.close()
    return sum(returns) / len(returns)


agent = Agent()

print(agent.policy)
print(agent.value_function)

state, info = env.reset()
reward = 0

t = 0
last_checkpoint = 0
transition_memory = TransitionMemory()
while True:
    transition_memory.clear()

    for i in range(trajectory_chunk_size):
        action = agent.get_action(state)

        state_new, reward, terminated, truncated, info = env.step(action)
        t += 1

        if terminated or truncated:
            reward = -10
            state_new, info = env.reset()

        transition_memory.append_transition(state, action, reward, state_new)
        state = state_new

    agent.update(transition_memory)
    writer.flush()

    if t // checkpoint_time != last_checkpoint:
        last_checkpoint = t // checkpoint_time
        average_return = test_agent(agent, 100, use_argmax=True)
        writer.add_scalar("Average Return", average_return, last_checkpoint)

        test_agent(agent, 1, show=True, use_argmax=True)
    
env.close()
writer.close()

