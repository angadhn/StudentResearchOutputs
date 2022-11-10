# PyTorch Imports
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.optim import Adam

# Other imports
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Project Imports
from networks import Recurrent_Actor
from networks import Critic

class RPPO:
    def __init__(self, env, continuous, save_path):
        """Init function

                Arguments:
                    env -- OpenAI Gym environment in which to train the network
                """

        self.init_hyperparameters()

        self.continuous = continuous
        self.prov_save_path = save_path
        self.actor_PATH = save_path + "\\actor.pth"
        self.critic_PATH = save_path + "\\critic.pth"
        self.plot_PATH = save_path + "\\rewards_plot.png"
        self.timesteps = []
        self.avg_rewards = []
        self.val_avg = []
        self.min_rewards = []
        self.val_min = []
        self.max_rewards = []
        self.val_max = []

        # extract environment information
        self.env = env
        self.state_dims = env.observation_space.shape[0]
        self.act_dims = env.action_space.shape[0]

        # initialize actor and critic networks
        self.actor = Recurrent_Actor(self.state_dims, self.act_dims, self.hidden_dims)
        self.critic = Critic(self.state_dims)

        # Initialize network optimizers
        self.ppo_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.action_var = torch.full(size=(self.act_dims,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.action_var)

    def init_hyperparameters(self):
        self.timesteps_per_batch = 5120
        self.max_timesteps_per_episode = 500
        self.gamma = 0.99
        self.lmbda = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.01
        self.hidden_dims = 64
        self.training_seeds = [20, 55, 137, 254, 329, 425, 567, 651, 739, 892, 946]
        self.val_seeds = [33, 77, 111, 222, 333, 444, 555, 666, 777, 888, 999]

    def learn(self, t_before_done):
        """Learn function

        Arguments:
            t_before_done -- number of timesteps to train the network for before stopping
        """

        start_time = time.time()

        t_so_far = 0  # number of timsteps simulated so far
        i = 0

        while t_so_far < t_before_done:

            i += 1

            iteration_start_time = time.time()

            # Get a batch of data of interactions between agent and environment
            batch_obvs, batch_acts, batch_log_probs, batch_rwds_tg, batch_rwds, batch_len, _ = self.rollout()
            # print("rollout done, average rnn loss: %.2f" % (torch.mean(batch_rnn_loss)))

            # Get value of observations V{phi, k}
            V, _ = self.evaluate(batch_obvs, batch_acts)

            # Calculate advantages A_k
            A = batch_rwds_tg - V.detach()
            # Normalize advantages
            A = (A - A.mean()) / (A.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate phi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obvs, batch_acts)

                # Calculate phis ratio
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate actor losses
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A
                # Calculate entire actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate the gradients and backpropagate actor loss through the actor network
                self.ppo_optim.zero_grad()
                actor_loss.backward()
                self.ppo_optim.step()

                # Calculate critic's MSE loss
                critic_loss = nn.MSELoss()(V, batch_rwds_tg)

                # Calculate the gradients and backpropagate critic loss through the critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            val_rewards = self.validate()

            # Calculated timesteps collected so far and time taken until now
            t_so_far += np.sum(batch_len)
            iteration_end_time = time.time()
            per_iteration_ptime = iteration_end_time - iteration_start_time
            total_time = iteration_end_time - start_time
            total_h, total_m, total_s = ((total_time / 60) / 60), ((total_time / 60) % 60), (total_time % 60)
            print("Iteration: %d || Total timesteps completed so far: %d  || Iteration completion time: %.2f s || Total time so far: %dh:%dm:%ds"
                % (i, t_so_far, per_iteration_ptime, total_h, total_m, total_s))

            # Report the results of the training iteration that has just finished
            ep_rwd_sum = [np.sum(t) for t in batch_rwds]
            print("TRAINING RESULTS: Average RWD-to-go: %.2f || Average batch reward: %.2f || Highest batch reward: %.2f || Lowest batch reward: %.2f" %
                  (torch.mean(batch_rwds_tg), np.mean(ep_rwd_sum), np.max(ep_rwd_sum), np.min(ep_rwd_sum)))
            val_rwd_sum = [np.sum(t) for t in val_rewards]
            print("VALIDATION RESULTS: Average validation reward: %.2f || Highest validation reward: %.2f || Lowest Validation reward: %.2f" %
                  (np.mean(val_rwd_sum), np.max(val_rwd_sum), np.min(val_rwd_sum)))
            print("Actor network loss: %.5f || Critic network loss: %.5f \n" %
                  (actor_loss, critic_loss))

            self.avg_rewards.append(np.mean(ep_rwd_sum))
            self.val_avg.append(np.mean(val_rwd_sum))
            self.min_rewards.append(np.min(ep_rwd_sum))
            self.val_min.append(np.min(val_rwd_sum))
            self.max_rewards.append(np.max(ep_rwd_sum))
            self.val_max.append(np.max(val_rwd_sum))
            self.timesteps.append(t_so_far)

            # Save the model and a plot of the rewards every 20 iterations
            # to allow for the testing of the model at different training stages
            if (i % 20) == 0:
                a_prov_path = self.prov_save_path + "\\actor_" + str(i) + ".pth"
                c_prov_path = self.prov_save_path + "\\critic_" + str(i) + ".pth"

                torch.save(self.actor.state_dict(), a_prov_path)
                torch.save(self.critic.state_dict(), c_prov_path)

                plt.plot(self.timesteps, self.avg_rewards, label="Average Training Episodic Return", color='blue')
                plt.fill_between(self.timesteps, self.max_rewards, self.min_rewards, facecolor='blue', alpha=0.5)

                plt.plot(self.timesteps, self.val_avg, label="Average Validation Return", color='red')
                plt.fill_between(self.timesteps, self.val_max, self.val_min, facecolor='red', alpha=0.5)

                plt.legend(loc='lower right')
                plt.xlabel('Timesteps')
                plt.ylabel('Episodic Return')

                plot_prov_path = self.prov_save_path + "\\prov_plot_" + str(i) + ".png"

                plt.savefig(plot_prov_path)
                plt.close()

        # Save the final version of the model and plots
        torch.save(self.actor.state_dict(), self.actor_PATH)
        torch.save(self.critic.state_dict(), self.critic_PATH)

        plt.plot(self.timesteps, self.avg_rewards, label="Average Training Episodic Return", color='blue')
        plt.fill_between(self.timesteps, self.max_rewards, self.min_rewards, facecolor='blue', alpha=0.5)

        plt.plot(self.timesteps, self.val_avg, label="Average Validation Return", color='red')
        plt.fill_between(self.timesteps, self.val_max, self.val_min, facecolor='red', alpha=0.5)

        plt.legend(loc='lower right')
        plt.xlabel('Timesteps')
        plt.ylabel('Episodic Return')

        plt.savefig(self.plot_PATH)
        plt.close()

    def rollout(self):
        """Rollout function

        Arguments:
            none
        """

        # batch data
        batch_obvs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probabilities of each action
        batch_rwds = []  # batch rewards
        batch_rwds_tg = []  # batch rewards to go
        batch_len = []  # episodic length in batch
        batch_state_val = []

        batch_tstps = 0

        while batch_tstps < self.timesteps_per_batch:

            ep_rwds = []  # rewards for specific episode
            ep_obvs = []
            ep_state_vals = []

            self.env.seed(random.choice(self.training_seeds))

            obvs = self.env.reset()
            hidden = self.actor.initHidden()
            done = False

            self.env.render()

            for ep_tstps in range(self.max_timesteps_per_episode):

                self.env.render()

                # increase batch timesteps
                batch_tstps += 1

                # collect observations
                ep_obvs.append(obvs)

                action, hidden, log_prob, state_val = self.get_action(obvs, hidden)
                obvs, rwd, done, _ = self.env.step(action)

                # collect reward, action and action log probability
                ep_rwds.append(rwd)
                ep_state_vals.append(state_val)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # collect the rewards and length of the episode
            batch_len.append(ep_tstps + 1)
            batch_rwds.append(ep_rwds)
            ep_state_vals.append(0)
            batch_state_val.append(ep_state_vals)
            batch_obvs.append(torch.tensor(ep_obvs, dtype=torch.float))

        # Reshape data as tensors in the shape specified before returning
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        advantages, batch_rwds_tg = self.compute_rwds_adv(batch_rwds, batch_state_val)

        # Return the batch data
        return batch_obvs, batch_acts, batch_log_probs, batch_rwds_tg, batch_rwds, batch_len, advantages

    def get_action(self, obvs, hidden):
        obvs = torch.tensor(obvs, dtype=torch.float)
        state_val = self.critic(obvs)

        # Query the actor network to get the action probabilities/mean
        obvs = obvs.unsqueeze(0).unsqueeze(0)
        action, hidden = self.actor(obvs, hidden)
        action = action.squeeze(0).squeeze(0)

        # Create action distribution
        # -- MultivariateNormal if the action space is continuous
        # -- Categorical if the action space is not continuous
        if self.continuous:
            dist = MultivariateNormal(action, self.cov_mat)
        else:
            dist = Categorical(action)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), hidden.detach(), log_prob.detach(), state_val.detach()

    def compute_rwds_adv(self, batch_rwds, batch_state_vals):
        # The rewards-to-go per episode per batch to return.
        # The shape will be (num timesteps per episode)
        advantages = []
        discounted_rewards = []

        for j in reversed(range(len(batch_rwds))):
            gae = 0  # The discounted reward so far
            discounted_reward = 0
            ep_rwds = batch_rwds[j]
            ep_state_vals = batch_state_vals[j]
            for i in reversed(range(len(ep_rwds))):
                delta = ep_rwds[i] + ep_state_vals[i + 1] * self.gamma - ep_state_vals[i]
                gae = delta + self.gamma * self.lmbda * gae
                advantages.insert(0, gae)

                discounted_reward = ep_rwds[i] + discounted_reward * 0.95
                discounted_rewards.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        advantages = torch.tensor(advantages, dtype=torch.float)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)

        return advantages, discounted_rewards

    def evaluate(self, batch_obvs, batch_acts):
        # Query the critic network for the value of each observation in a batch of observations
        input = torch.cat(batch_obvs, dim=0)
        V = self.critic(input).squeeze()

        actions = []

        # Calculate the log probabilities of a batch of actions using the most recent actor network
        # Query the actor network to get the action probabilities/mean
        for obvs in batch_obvs:
            hidden = self.actor.initHidden()
            obvs = obvs.unsqueeze(0)
            action, hiddens = self.actor(obvs, hidden)

            actions.append(action.squeeze(0))

        actions = torch.cat(actions, dim=0)

        # Create action distribution
        # -- MultivariateNormal if the action space is continuous
        # -- Categorical if the action space is not continuous
        if self.continuous:
            dist = MultivariateNormal(actions, self.cov_mat)
        else:
            dist = Categorical(actions)

        # Get the log probabilities of the batch of actions
        log_probs = dist.log_prob(batch_acts)
        #entropy = dist.entropy()

        return V, log_probs

    def validate(self):
        # batch data
        batch_rwds = []  # batch rewards

        batch_tstps = 0

        while batch_tstps < self.timesteps_per_batch:

            ep_rwds = []  # rewards for specific episode

            self.env.seed(random.choice(self.val_seeds))

            obvs = self.env.reset()
            hidden = self.actor.initHidden()
            done = False

            self.env.render()

            while not done:

                self.env.render()

                # increase batch timesteps
                batch_tstps += 1

                obvs = torch.tensor(obvs, dtype=torch.float)
                obvs = obvs.unsqueeze(0).unsqueeze(0)
                action, hidden = self.actor(obvs, hidden)
                action = action.squeeze(0).squeeze(0)
                obvs, rwd, done, _ = self.env.step(action.detach().numpy())

                # collect reward, action and action log probability
                ep_rwds.append(rwd)

            # collect the rewards and length of the episode
            batch_rwds.append(ep_rwds)

        # Return the validation rewards
        return batch_rwds