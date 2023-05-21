#!/usr/bin/env python3
import argparse
import numpy as np
import gym
from gym_minigrid.wrappers import *
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch
import time
import os


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)


def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    print(env.observation_space, env.action_space.n)
    print(env.observation_space['image'])
    print(obs['image'])

    if done:
        print('done!')
        reset()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="gym environment to load",
                        default='MiniGrid-Empty-5x5-v0')
    parser.add_argument("--seed", type=int,
                        help="random seed to generate the environment with",
                        default=-1)
    parser.add_argument('--lr', default=1e-4, help="learning rate",)
    parser.add_argument('--epoch_number', default=100, help="train epoches")
    parser.add_argument('--collect_number', default=10,
                        help="the number of trajectories of each epoch")
    parser.add_argument('--alpha', default=0.1, type=float, help="alpha")
    parser.add_argument('--sample_algorithm', default="IS", type=str)
    parser.add_argument('--use_biris', default=False, type=bool)
    parser.add_argument('--buffer_size', default=40, type=int)

    return parser.parse_args()


class Replay_Buffer():
    def __init__(self):
        self.len = 0
        self.obs = []
        self.acts = []
        self.rews = np.zeros(0)
        self.logps = np.zeros(0)

    def add_item(self, obss, actss, rew, logp):
        self.len += 1
        self.obs.append(obss)
        self.acts.append(actss)
        self.rews = np.append(self.rews, [rew])
        self.logps = np.append(self.logps, [logp])


class Agent(nn.Module):
    def __init__(self, env, alpha=0.1):
        super().__init__()
        # env.observation_space: Dict(image:Box(7, 7, 3))
        n = env.observation_space["image"].shape[0]
        m = env.observation_space["image"].shape[1]

        action_space = env.action_space.n
        # image input: n * m * 3
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.embedding_size = ((n-1)//2 - 2) * ((m-1)//2 - 2) * 64

        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )

        self.alpha = alpha
        print('alpha is:', alpha)

    def forward(self, obs):
        # print(obs.shape)
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.actor(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist

    def get_action(self, obs):
        obs = [ob["image"] for ob in obs]
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = self(obs)
        actions = dist.sample()
        return actions.cpu().numpy(), dist.logits[0][actions].cpu().numpy()

    def sample_traj(self, env, num=1, replay_buffer=None, store=False):
        obs = env.reset()
        total_reward = 0.0
        total_logp = 0.0
        obss = []
        actss = []
        traj_num = 0
        average_reward = 0.0
        while traj_num < num:
            action, logp = agent.get_action([obs])
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            total_logp += logp
            obss.append(obs)
            actss.append(action)
            if done:
                average_reward += total_reward
                if replay_buffer is not None and store == True:
                    replay_buffer.add_item(obss, actss, total_reward, total_logp)
                next_obs = env.reset()
                total_reward = 0.0
                total_logp = 0.0
                obss = []
                actss = []
                traj_num += 1
            obs = next_obs

        average_reward /= num
        print("average reward is:", average_reward)
        return average_reward

    def update(self, replay_buffer, optimizer):
        for _ in range(20):
            loss1 = torch.tensor([0.0], dtype=torch.float32)
            loss2 = torch.tensor([0.0], dtype=torch.float32)
            total_logp = torch.tensor([0.0], dtype=torch.float32)
            for i in range(replay_buffer.len):
                obss = replay_buffer.obs[i]
                acts = replay_buffer.acts[i]
                logp = -torch.tensor(replay_buffer.logps[i], dtype=torch.float32)
                reward = torch.tensor([replay_buffer.rews[i]], dtype=torch.float32)
                for j in range(len(obss)):
                    obs = obss[j]
                    obs = [obs["image"]]
                    obs = np.array(obs)
                    obs = torch.tensor(obs, dtype=torch.float32)
                    dist = self(obs)
                    logp = logp + dist.logits[0][acts[j]]
                # print(logp)
                # print(reward)
                loss1 -= torch.exp(logp) * reward
                total_logp += torch.exp(logp)

                # add biris
                loss2 += self.alpha * torch.abs(torch.exp(logp) - 1)
            if args.sample_algorithm == 'WIS':
                loss1 = loss1 / total_logp
            else:
                loss1 = loss1 / replay_buffer.len
            print("estimated reward", -loss1.item())

            loss2 = loss2 / replay_buffer.len

            if args.use_biris:
                loss = loss1 + loss2
            else:
                loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        estimated_reward = torch.tensor([0.0], dtype=torch.float32)
        total_logp = torch.tensor([0.0], dtype=torch.float32)
        for i in range(replay_buffer.len):
            obss = replay_buffer.obs[i]
            acts = replay_buffer.acts[i]
            logp = -torch.tensor(replay_buffer.logps[i], dtype=torch.float32)
            reward = torch.tensor([replay_buffer.rews[i]], dtype=torch.float32)
            for j in range(len(obss)):
                obs = obss[j]
                obs = [obs["image"]]
                obs = np.array(obs)
                obs = torch.tensor(obs, dtype=torch.float32)
                dist = self(obs)
                logp = logp + dist.logits[0][acts[j]]
            # print(logp)
            # print(reward)
            estimated_reward += torch.exp(logp) * reward
            total_logp += torch.exp(logp)
        if args.sample_algorithm == 'WIS':
            estimated_reward = estimated_reward / total_logp
        else:
            estimated_reward = estimated_reward / replay_buffer.len
        print("estimated reward", estimated_reward.item())
        return estimated_reward.item()


if __name__ == '__main__':
    args = get_args()
    path = "./data/" + args.env + "/" + args.sample_algorithm
    print('path:', path)
    os.makedirs(path, exist_ok=True)

    env = gym.make(args.env)
    print(env.observation_space, env.action_space.n)

    estimated_return = []
    average_return = []
    for seed_id in range(50):
        start_time = time.time()
        args.seed = seed_id
        print("seed:", seed_id)
        replay_buffer = Replay_Buffer()
        agent = Agent(env, alpha=args.alpha)
        optim = torch.optim.Adam(agent.parameters(), lr=args.lr)
        _ = agent.sample_traj(env, args.buffer_size, replay_buffer, True)
        estimated_return.append(agent.update(replay_buffer, optim))
        average_return.append(agent.sample_traj(env, 100))
        print("Time cost:", time.time() - start_time)
    estimated_return = np.array(estimated_return)
    average_return = np.array(average_return)
    print(estimated_return)
    print(estimated_return.mean())
    print(average_return)
    print(average_return.mean())
    print((estimated_return.mean() - average_return.mean()) / average_return.mean())

    if args.use_biris:
        np.save("./data/" + args.env + "/" +
                args.sample_algorithm + "/biris_estimated_" +
                str(args.buffer_size) + "_" + str(args.alpha) + ".npy",
                estimated_return)
        np.save("./data/" + args.env + "/" +
                args.sample_algorithm + "/biris_average_" +
                str(args.buffer_size) + "_" + str(args.alpha) + ".npy",
                average_return)
    else:
        np.save("./data/" + args.env + "/" +
                args.sample_algorithm + "/estimated_" +
                str(args.buffer_size) + ".npy", estimated_return)
        np.save("./data/" + args.env + "/" +
                args.sample_algorithm + "/average_" +
                str(args.buffer_size) + ".npy", average_return)