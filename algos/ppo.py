import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from common.buffers import TrajectoryBuffer
from gym import Env
from utils.train_tools import soft_target_update, explore_before_train, evaluate
from utils import log_tools


class PPO_Agent:
    """
    Implementation of Proximal Policy Optimization (PPO)
    This is the version of "PPO-Clip"
    https://arxiv.org/abs/1707.06347
    """
    def __init__(self,
                 env: Env,
                 trajectory_buffer: TrajectoryBuffer,
                 actor_net: torch.nn.Module,
                 critic_net: torch.nn.Module,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 gae_normalize=False,
                 clip_pram=0.2,
                 trajectory_length=128,  # the length of a trajectory_
                 train_actor_iters=10,
                 train_critic_iters=10,
                 eval_freq=1000,   # it will not evaluate the agent during train if eval_freq < 0
                 max_time_step=10000,
                 train_id="PPO_CarPole_test",
                 log_interval=1000,
                 resume=False,  # if True, train from last checkpoint
                 device='cpu'):
        self.env = env

        self.trajectory_buffer = trajectory_buffer

        self.device = torch.device(device)

        # the network and optimizers
        self.actor_net = actor_net.to(self.device)
        self.critic_net = critic_net.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.gae_normalize = gae_normalize
        self.trajectory_length = trajectory_length
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.clip_pram = clip_pram

        self.eval_freq = eval_freq
        self.max_time_step = max_time_step

        self.time_step = 0
        self.episode_num = 0

        self.resume = resume  # whether load checkpoint start train from last time

        # log dir and interval
        self.log_interval = log_interval
        self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", train_id)
        log_tools.make_dir(self.result_dir)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
        self.tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

    def choose_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
            action, log_prob, eval_action = self.actor_net(obs)
            if eval:
                action = eval_action
        return action.cpu().numpy().squeeze(0), log_prob.cpu().numpy()[0]

    def train(self):
        batch = self.trajectory_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        log_probs = batch["log_probs"].to(self.device)
        gae_advs = batch["gae_advs"].to(self.device)
        rets = batch["rets"].to(self.device)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            _, new_log_probs, _ = self.actor_net(obs, acts.squeeze())
            ratios = torch.exp(new_log_probs - log_probs)

            surrogate = ratios * gae_advs
            clipped_surrogate = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * gae_advs
            actor_loss = -(torch.min(surrogate, clipped_surrogate)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Train value function with multiple steps of gradient descent
        for i in range(self.train_critic_iters):
            values = self.critic_net(obs).squeeze()
            critic_loss = 0.5 * ((rets - values) ** 2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        return actor_loss.cpu().item(), critic_loss.cpu().item()

    def learn(self):
        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)
        print("==============================start train===================================")
        obs = self.env.reset()
        done = False

        episode_reward = 0
        episode_length = 0
        trajectory_length = 0

        while self.time_step < self.max_time_step:
            action, log_prob = self.choose_action(np.array(obs))
            next_obs, reward, done, info = self.env.step(action)
            value = self.critic_net(torch.tensor([obs], dtype=torch.float32)).item()
            episode_reward += reward
            self.trajectory_buffer.add(obs, action, reward, done, log_prob, value)
            obs = next_obs
            episode_length += 1
            trajectory_length += 1
            self.time_step += 1

            if done:
                obs = self.env.reset()
                self.episode_num += 1

                print(f"Time Step: {self.time_step} Episode Num: {self.episode_num} "
                      f"Episode Length: {episode_length} Episode Reward: {episode_reward:.3f}")
                self.tensorboard_writer.log_learn_data({"episode_length": episode_length,
                                                        "episode_reward": episode_reward}, self.time_step)
                episode_reward = 0
                episode_length = 0

            if trajectory_length == self.trajectory_length:
                last_val = self.critic_net(torch.tensor([obs], dtype=torch.float32)).item() if done else 0
                self.trajectory_buffer.finish_path(last_val=last_val, gamma=self.gamma,
                                                   gae_lambda=self.gae_lambda, gae_normalize=self.gae_normalize)
                actor_loss, critic_loss = self.train()
                trajectory_length = 0

            if self.time_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                self.tensorboard_writer.log_train_data({"actor_loss": actor_loss,
                                                        "critic_loss": critic_loss}, self.time_step)

            if self.eval_freq > 0 and self.time_step % self.eval_freq == 0:
                avg_reward, avg_length = evaluate(agent=self, episode_num=10)
                self.tensorboard_writer.log_eval_data({"eval_episode_length": avg_length,
                                                       "eval_episode_reward": avg_reward}, self.time_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "time_step": self.time_step,
            "episode_num": self.episode_num
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.time_step = checkpoint["time_step"]
        self.episode_num = checkpoint["episode_num"]
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.time_step) + " time step")








