import numpy as np
import torch
from algos.base import PolicyBase
from common.buffers import TrajectoryBuffer
from utils.train_tools import evaluate
from utils import log_tools


class PPO_Agent(PolicyBase):
    """
    Implementation of Proximal Policy Optimization (PPO)
    This is the version of "PPO-Clip"
    https://arxiv.org/abs/1707.06347
    """
    def __init__(self,
                 trajectory_buffer: TrajectoryBuffer,
                 actor_net: torch.nn.Module,
                 critic_net: torch.nn.Module,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gae_lambda=0.95,
                 gae_normalize=False,
                 clip_pram=0.2,
                 trajectory_length=128,  # the length of a trajectory_
                 train_actor_iters=10,
                 train_critic_iters=10,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.trajectory_buffer = trajectory_buffer

        # the network and optimizers
        self.actor_net = actor_net.to(self.device)
        self.critic_net = critic_net.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        self.gae_lambda = gae_lambda
        self.gae_normalize = gae_normalize
        self.trajectory_length = trajectory_length
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.clip_pram = clip_pram

        self.episode_num = 0

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

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item()}

        return train_summaries 

    def learn(self):
        log_tools.make_dir(self.result_dir)
        tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

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

        while self.train_step < self.max_train_step:
            action, log_prob = self.choose_action(np.array(obs))
            next_obs, reward, done, info = self.env.step(action)
            value = self.critic_net(torch.tensor([obs], dtype=torch.float32)).item()
            episode_reward += reward
            self.trajectory_buffer.add(obs, action, reward, done, log_prob, value)
            obs = next_obs
            episode_length += 1
            trajectory_length += 1
            self.train_step += 1

            if done:
                obs = self.env.reset()
                self.episode_num += 1

                print(f"Time Step: {self.train_step} Episode Num: {self.episode_num} "
                      f"Episode Length: {episode_length} Episode Reward: {episode_reward:.2f}")
                tensorboard_writer.log_learn_data({"episode_length": episode_length,
                                                        "episode_reward": episode_reward}, self.train_step)
                episode_reward = 0
                episode_length = 0

            if trajectory_length == self.trajectory_length:
                last_val = self.critic_net(torch.tensor([obs], dtype=torch.float32)).item() if done else 0
                self.trajectory_buffer.finish_path(last_val=last_val, gamma=self.gamma,
                                                   gae_lambda=self.gae_lambda, gae_normalize=self.gae_normalize)
                train_summaries = self.train()
                trajectory_length = 0

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                tensorboard_writer.log_train_data(train_summaries, self.train_step)

            if self.eval_freq > 0 and self.train_step % self.eval_freq == 0:
                evaluate_summaries = evaluate(agent=self, episode_num=10)
                tensorboard_writer.log_eval_data(evaluate_summaries, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "train_step": self.train_step,
            "episode_num": self.episode_num
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")
