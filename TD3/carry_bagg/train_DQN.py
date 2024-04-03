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
from velodyne_env import GazeboEnv



# Crea el entorno de entrenamiento 
environment_dim = 13
seed = 0  # Random seed number
robot_dim = 2
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)



####################################################3333
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#######################################################
#Memoria Buffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','done','next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



#######################################################



#RED NEURONAL DQN with 3 layers 

class netDQN(nn.Module):
    

    def __init__(self, state_dim, action_dim):
        super(netDQN, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        
    #Regresa un tensor a con la acción más adecueda
    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.layer_3(s)
        return a
      
#################################################################################

#Parámetros de la implementación
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
print(device)
batch_size = 40  # Size of the mini-batch
gamma = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
#Epsilo greedy exploration
epsilonStart = .99  # Initial exploration noise starting value in range [expl_min ... 1]
epsilonDecay = 190  # Number of steps over which the initial exploration noise will decay over
epsilonMin = 0.1  # Exploration noise after the decay in range [0...expl_noise]
#Tau es el factor de actualización de la red Target
tau = 0.005  # Soft target update variable (should be close to 0)
#Factor de aprendizaje AdamW Optimizer
alfa=1e-4
max_ep = 400  # maximum number of steps per episode
REPLAY_START_SIZE = 10000


#########################################################################

#Obtener el número de dimensiones y acciones
state_dim = environment_dim + robot_dim
# Se tendrán 3 acciones, Avanzar, Derecha, Izquierda
action_dim = 3
action_space = [0,1,2] #  Avanzar, Derecha, Izquierda



#########################################################################3

eval_freq = 25  # After how many steps to perform the evaluation
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e4  # Maximum number of steps to perform
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e5  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")


################################################33
#ENTRENAMIENTO 

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = len(action_space)
print("Numero de acciones:" + str(n_actions))
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = netDQN(n_observations, n_actions).to(device)
target_net = netDQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0

####Selecciona una acción
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    print("Epsilon Greedy" + str(eps_threshold))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        action = random.sample(action_space,1)
        return torch.tensor([action], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(episode_list,average_rewards_per_step,show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Media acumulada de recompensas ')
    plt.plot(episode_list, average_rewards_per_step, marker='o', linestyle='-')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

##############################################################################################            


###############################################################################################

#Optimización model

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #print(action_batch)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #print(policy_net(state_batch))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

##################################################################################3

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 100

# Nombre del archivo de texto
archivo_texto = "variables.txt"
# Abre el archivo en modo escritura



average_rewards_per_step = []
episode_list = []

#Carga pesos de la red
policy_net.load_state_dict(torch.load('modelo.pth'))
#policy_net.eval() 

for i_episode in range(num_episodes):
    episode_timesteps=0
    averageRewards = 0
    
    # Initialize the environment and get its state
    state = env.reset()
    
    print("Epsiodio" + str(i_episode))
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(state.shape)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, _ = env.step(action.item())
        
        reward = torch.tensor([reward], device=device)
        done = terminated 

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, reward,done,next_state)

        # Move to the next state
        state = next_state

        

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        

        episode_timesteps = episode_timesteps + 1
        #print("Reward: " + str(float(reward)))
        averageRewards=averageRewards + float(reward) 
        
        print("Time Steps:" + str(episode_timesteps)+","+"Rewards" + str(averageRewards))


        if done:
            episode_durations.append(t + 1)
            averageRewards = averageRewards / episode_timesteps
            plot_durations(episode_list,average_rewards_per_step,False)
            break
     
        with open(archivo_texto, "a") as archivo:
              # Escribe el estado de las variables en el archivo
             archivo.write("{},".format(state))
             archivo.write("{} \n".format(action))
       
    
    #print(averageRewards)
    average_rewards_per_step.append(averageRewards)
    episode_list.append(i_episode)
    #print(average_rewards_per_step)
    #print(episode_list)
    #print(i_episode)
    # Graficar los promedios de recompensa por paso
    plt.plot(episode_list, average_rewards_per_step, marker='o', linestyle='-')
    plt.title('Promedio de Recompensa por Paso')
    plt.xlabel('Episodio')
    plt.ylabel('Promedio de Recompensa')
    plt.grid(True)
    plt.show()

print('Complete')
#plot_durations(show_result=F)
#plt.ioff()
#plt.show()


# Guarda el modelo
torch.save(policy_net.state_dict(), 'modelo.pth')
# Guarda el modelo
print("Modelo guardado")

##########################################################
#Experimentando con aproximación multivariada



##########################################################3





