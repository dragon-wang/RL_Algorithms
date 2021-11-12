import copy
import os
import numpy as np
import torch
from common.buffers import ReplayBuffer
from gym import Env
from utils.train_tools import hard_target_update, explore_before_train
from utils import log_tools


class DQN_Agent:
    """
    Implementation of Deep Q-Network (DQN)
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    def __init__(self,
                 env: Env,
                 replay_buffer: ReplayBuffer,
                 Q_net: torch.nn.Module,
                 qf_lr=0.001,
                 gamma=0.99,
                 initial_eps=0.1,
                 end_eps=0.001,
                 eps_decay_period=2000,
                 eval_eps=0.001,
                 target_update_freq =10,
                 train_interval: int = 1,
                 explore_step=500,
                 max_train_step=10000,
                 train_id="dqn_CartPole_test",
                 log_interval=1000,
                 resume=False,  # if True, train from last checkpoint
                 device='cpu'
                 ):
        self.env = env
        self.replay_buffer = replay_buffer

        self.explore_step = explore_step
        self.max_train_step = max_train_step
        self.train_interval = train_interval
        self.target_update_freq = target_update_freq

        self.device = torch.device(device)

        self.Q_net = Q_net.to(self.device)
        self.target_Q_net = copy.deepcopy(self.Q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=qf_lr)

        # Decay for epsilon
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        self.eval_eps = eval_eps

        self.gamma = gamma
        self.train_step = 0
        self.episode_num = 0

        self.resume = resume  # whether load checkpoint start train from last time

        # log dir and interval
        self.log_interval = log_interval
        self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", train_id)
        log_tools.make_dir(self.result_dir)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
        self.tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

    def choose_action(self, obs, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.train_step + self.initial_eps, self.end_eps)

        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                return int(self.Q_net(obs).argmax(dim=1).cpu())
        else:
            return self.env.action_space.sample()

    def train(self):
        """
        Sample a batch of data from replay buffer and train
        """

        # Sample
        batch = self.replay_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Compute target Q value
        with torch.no_grad():
            target_q = rews + (1. - done) * self.gamma * self.target_Q_net(next_obs).max(dim=1)[0]

        # Compute current Q value
        current_q = self.Q_net(obs).gather(1, acts.long()).squeeze(1)

        # Compute Q loss
        q_loss = 0.5 * (target_q - current_q).pow(2).mean()
        # Q_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        self.train_step += 1

        # update target Q
        if self.train_step % self.target_update_freq == 0:
            hard_target_update(self.Q_net, self.target_Q_net)

        return q_loss.cpu().item()

    def learn(self):
        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)
        explore_before_train(self.env, self.replay_buffer, self.explore_step)
        print("==============================start train===================================")
        obs = self.env.reset()
        done = False

        episode_reward = 0
        episode_length = 0
        q_loss = 0

        while self.train_step < self.max_train_step:
            action = self.choose_action(np.array(obs))
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_length += 1

            if (self.train_step+1) % self.train_interval == 0:
                q_loss = self.train()

            if done:
                self.episode_num += 1
                obs = self.env.reset()
                done = False

                print(f"Total T: {self.train_step} Episode Num: {self.episode_num} "
                      f"Episode Length: {episode_length} Episode Reward: {episode_reward:.3f}")
                self.tensorboard_writer.log_learn_data({"episode_length": episode_length,
                                                        "episode_reward": episode_reward}, self.train_step)

                episode_reward = 0
                episode_length = 0

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                self.tensorboard_writer.log_train_data({"Q_loss": q_loss}, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "net": self.Q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "episode_num": self.episode_num
        }
        torch.save(checkpoint, self.checkpoint_path)
        # print("checkpoint saved in " + self.checkpoint_path + " at " + str(self.train_step+1) + " step")

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.Q_net.load_state_dict(checkpoint["net"])
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]
        print("load checkpoint from " + self.checkpoint_path +
              " and start train from " + str(self.train_step+1) + "step")