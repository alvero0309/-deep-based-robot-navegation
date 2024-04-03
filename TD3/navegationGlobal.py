#The navegation complete needs the navegation local

#It is based on optimal points that were obtained from the environment with 2D lidar.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import collections
import random
import math
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from numpy import inf
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter
#Es necesario un replay Buffer
from replay_buffer import ReplayBuffer
from laser_env import GazeboEnv

#Creates the array of the Laser 
environment_dim = 15
seed = 0  # Random seed number
#The robot dim is the velocity angular and velocity linear
robot_dim = 2
#This is the environment for training the navegation complete
env = GazeboEnv("complete_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

#The global navegation is performed for the navegation trough of nodes 

#First: I will get the goal position in terms of degree relative to the robot and coordinates in x and y dimensions

# Initialize the environment and get its state(The state is the postion goal and raw laser)
state = env.reset()
#Maps the environment and put points of interest


episode_timesteps=0
averageRewards = 0
    
    
    
