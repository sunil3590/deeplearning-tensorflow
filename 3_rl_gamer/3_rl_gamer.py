#!/usr/bin/python

# Reference - https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
from gym import wrappers
import time


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


class Gamer:
    def __init__(self):
        # set up the environment
        self.env = gym.make("CartPole-v1")
        self.folder = './' + str(int(round(time.time())))
        self.env = wrappers.Monitor(self.env, self.folder)
        # rl agent
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # history of rewards and actions for an episode
        self.actions = list()
        self.rewards = list()

    # identify next action and remember it
    def _next_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self.model(Variable(observation))
        action = probs.multinomial()
        self.actions.append(action)
        return action.data[0, 0]

    def _learn(self):
        disc_factor = 0.99
        discounted_future_reward = 0
        discounted_rewards = []
        for r in self.rewards[::-1]:
            discounted_future_reward = r + disc_factor * discounted_future_reward
            discounted_rewards.insert(0, discounted_future_reward)
        discounted_rewards = torch.Tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() +
                                                                                 np.finfo(np.float32).eps)
        for action, discounted_reward in zip(self.actions, discounted_rewards):
            action.reinforce(discounted_reward)
        self.optimizer.zero_grad()
        autograd.backward(self.actions, [None for _ in self.actions])
        self.optimizer.step()

    def _reset_stuff(self):
        self.actions = list()
        self.rewards = list()

    def play(self, n_episodes, n_render):
        # reset the environment to start
        observation = self.env.reset()

        # run the experiment over requested episodes
        episode = 0
        while n_episodes > episode:
            # render if requested
            if n_render > 0 and episode % n_render == 0:
                self.env.render()

            # choose next action based on observation
            action = self._next_action(observation)

            # execute next action and remember reward
            observation, reward, done, info = self.env.step(action)
            self.rewards.append(reward)

            # episode complete - learn and reset stuff
            if done:
                print "Episode", episode, " Score", np.sum(self.rewards)
                self._learn()
                self._reset_stuff()
                observation = self.env.reset()
                episode += 1

    def upload_to_gym(self):
        self.env.close()
        gym.upload(self.folder, api_key='')


def main():
    gamer = Gamer()
    gamer.play(5000, -1)  # run 5000 episodes and do not render
    gamer.upload_to_gym()

if __name__ == "__main__":
    main()
