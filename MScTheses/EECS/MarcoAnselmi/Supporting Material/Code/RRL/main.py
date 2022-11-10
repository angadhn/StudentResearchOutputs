import gym
import os
import torch

from RPPO import RPPO
from networks import Recurrent_Actor
from eval import rollout
from tester import tester

# Create the Gym environment
lunar_lander_v2 = gym.make('LunarLanderContinuous-v2')

# Create Folders to save models and plots
cwd = os.getcwd()
save_dir = os.path.join(cwd, 'Results')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train():
    # Run model
    model = RPPO(lunar_lander_v2, True, save_dir)
    model.learn(250_000)

def test():
    model = Recurrent_Actor(lunar_lander_v2.observation_space.shape[0], lunar_lander_v2.action_space.shape[0], 64)
    model_path = os.path.join(cwd, "Results\\actor.pth")
    model.load_state_dict(torch.load(model_path))
    tester(model, lunar_lander_v2)

def eval():
    model = Recurrent_Actor(lunar_lander_v2.observation_space.shape[0], lunar_lander_v2.action_space.shape[0], 64)
    model_path = os.path.join(cwd, "Results\\actor.pth")
    model.load_state_dict(torch.load(model_path))
    rollout(model, lunar_lander_v2)

if __name__ == '__main__':
    t = 2
    if t == 0:
        eval()
    elif t == 1:
        train()
    elif t == 2:
        test()


'''
The code for the training of the PPO algorithm has been taken from the tutorial written by Eric Yang Yu 

The tutorial can be found at the following link: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
'''