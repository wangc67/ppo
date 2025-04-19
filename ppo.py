import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class PPOConfig:
    ACTOR_LR = 0.1
    CRITIC_LR = 0.1
    GAMMA = 0.99
    LAMBDA = 0.99
    EPS_CLIP = 0.5

    EPOCHS = 25
    NUM_EPISODE = 500
    SEED = None

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # return mean and var of Gaussian prob of continuous actions
        mu, std = 0, 1
        return mu, std

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def compute_advantage(gamma, lmbda, td_delta):
    # generalized advantage estimation
    td_delta = td_delta.detach().numpy()
    advantage_lst = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_lst.append(advantage)
    advantage_lst.reverse()
    return torch.tensor(advantage_lst)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 actor_lr = PPOConfig.ACTOR_LR, 
                 critic_lr = PPOConfig.CRITIC_LR,
                 lmbda = PPOConfig.LAMBDA,
                 epochs = PPOConfig.EPOCHS,
                 eps = PPOConfig.EPS_CLIP,
                 gamma = PPOConfig.GAMMA,
                 device = 'cpu'
                 ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def save_model(self):
        pass

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        # td_delta = td_target - self.critic(states)
        

        
        # advantage = compute_advantage()
        
        # mu, std = self.actor(states)
        # action_dists = torch.distributions.Normal(mu.detach(),std.detach())

        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)

            log_probs = action_dists.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage # PPO-CLIP
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def train_on_policy_agent(env, agent, num_episodes=PPOConfig.NUM_EPISODE):
    return_lst = []
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        transition_dict = {'states':[], 'actions':[], 'next_states':[], 'rewards':[], 'dones':[]}
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            
            transition_dict['actions'].append(action)
            transition_dict['states'].append(state)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
        
        if i_episode % 20 == 0:
            agent.save_model()


        return_lst.append(episode_return)
        agent.update(transition_dict)

    return return_lst

def workflow(envs, agents):
    do_learns = [True, True]
    while True:
        for g_data in run_episodes():
            for idx, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                agent.learn(g_data[idx])
            g_data.clear()

            if condition:
                agent[0].save_model()


def run_episode(env, agent):
    # self play

    train_agent_id = 0
    while True:
        # train_agent_id = 1 - train_agent_id
        opponent_agent = 'selfplay'
        if is_eval:
            pass

        _, state_dicts = env.reset()
        
        # reset agent 
        for i, agent in enumerate(agents):
            player_id = 0
            camp = 0
            agent.reset()

            if i == train_agent_id:
                agent.load_model('latest')
            else:
                if opponent_agent == 'selfplay':
                    agent.load_model('latest')
        # reset reward
        for i in range(agent_num):
            reward = agents[i].reward_manager_reset('?')
        
        while True:
            for idx, d_predict, aagent in enumerate(zip(do_predictions, agents)):
                if d_predict:
                    if not is_eval:
                        actions[idx] = agent.train_predict()
                    else:
                        actions[idx] = agent.eval_predict()

            transition_dict = env.step(actions)

            for i in range(agent_num):
                compute_reward()

            if terminated or truncated:
                for idx, d_predict,agent in enumerate(zip(do_predicts, agents)):    
                    if d_predict and not is_eval:
                        save_reward()


if __name__ == '__main__':
    env = Game()
    agent = PPO()
    return_lst = train_on_policy_agent(env, agent)
    exit()
