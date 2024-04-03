import os
import time

import numpy as np
from numpy import inf
import random
#from torch.utils.tensorboard import SummaryWriter
import math


from tensorboardX import SummaryWriter

#Discretización de acciones
# Crea el entorno de entrenamiento
seed=0
#dimension de los estados de entrada, el barrido es de 180° con respecto el frente del robot
environment_dim = 20
#El espacio de estados del robot esta dado por la distancia de la meta y el angulo con respecto el robot, 
#la velocidad lineal y angular del robot
robot_dim = 4
#Se inicializa el multiescenario, en gazebo
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

#Dimension de los estados será una discretización de la nube de puntos más la información del robot
state_dim = environment_dim + robot_dim #24
#Velocidad lineal y velocidad angular
action_dim = 2
#Limite de la maxíma acción
max_action = 10

#Espacio de aciones - Definir Giros
Turn0=0
TurnLeft45=math.pi/4
TurnRight45=-(math.pi/4)
TurnRight90 = -(math.pi/2)
TurnLeft90 = (math.pi/2)

#Espacio de accines 
#action_dim=[Forward(),TurnRight(TurnRight45),TurnRight(TurnRight90),TwistLeft(TurnLeft45),TwistLeft(TurnLeft90)]
#Detectar la orientación de la fuente luminosa y discretizarla a las direcciones, Norte=90°, Sur = 270°, Oeste180°, este 0°,Noroeste 135°
#Noreste 45°, suereste 225° suroesteste 315°

#Verificar angulo de acción

def discreteLight(stateLight):

    #Este (-22.5° a 22.5°)
    if (stateLight > -0.3926991) and (stateLight < 0.3926991):
      degree = 0 # o pi
    #Noreste  (22.6 a 67.49)
    if (stateLight > 0.3926991) and (stateLight < 1.178097):
      degree = 0.785398 # pi/4
    #Norte  (67.49 - 112.5)
    if (stateLight > 1.178097) and (stateLight < 1.9634954):
      degree = 1.5708 # pi/2
     #Noroeste  (112.5 a 157.5)
    if (stateLight > 1.9634954) and (stateLight < 2.7488936):
      degree = 2.7488936# pi 135°
     #Oeste  (157.5 a 180)
    if (stateLight > 2.7488936) and (stateLight <= 3.14159):
      degree = 3.14159# pi 


    #Sureste  (22.6 a 67.49)
    if (stateLight < -0.3926991) and (stateLight > -1.178097):
      degree = -0.785398 # pi/4
    #Norte  (67.49 - 112.5)
    if (stateLight < -1.178097) and (stateLight > -1.9634954):
      degree = -1.5708 # pi/2
     #Noroeste  (112.5 a 157.5)
    if (stateLight < -1.9634954) and (stateLight > -2.7488936):
      degree = -2.7488936# pi 135°
     #Oeste  (157.5 a 180)
    if (stateLight < -2.7488936) and (stateLight >= -3.14159):
      degree = -3.14159# pi 

    return degree





#Discretización de los estados
#Cada medida de la nube de puntos esta discretizada con el objeto más cercano con respecto al robot, dividida en 
#180/20, cada 9° se encuentra medido el punto minimo con respecto si se encuentra un obstaculo, es necesario definir un umbral
#para definir cuando es un espacio ocupado, en este caso el valor maximo 5 y el minimo es 3
#Si no existe algún obstaculo 10 es la medida del lidar 
def stateDiscrete(state):
    #Además de las lecturas laser, se encuentra concatenado el estado del robot con respecto al objetivo
    done=0
    stateD = []
    #Extrae el estado del robot
    stateR = state[-4:]
    #Extrae la distancia y el angulo del objetivo con respecto al robot
    stateLight = stateR[:-2]
    #Se extraen las lecturas laser (20)
    stateLaser= state[:-4]
    
    #Angulos discretos 8 # La función de argumento solo necesita el angulo
    angleDiscrete=discreteLight(stateLight[1])
    distanceTarget=stateLight[0]
    for i in range(len(stateLaser)):
        #No existe obstaculo
        if stateLaser[i] >= 3:
          stateD.append(0)
        else:
        #Existe obstaculo
          stateD.append(1)

    if distanceTarget < .3:
       done = 1

    
    stateD.append(angleDiscrete)
    #Anexa estado terminal
    stateD.append(done)

    #Agregar estado terminal

    return stateD







#Discretización del espacio de acciones 

#Avanzar una cierta cantidad de pasos


#Se tendrán 3 acciones en su max velocidad Avanzar, girar Izquierda, girar Derecha
def Forward():
  action=[1,0]
   # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
  a_in = [(action[0] + 1) / 2, action[1]]
  infoEnvironment = env.step(a_in)
  return infoEnvironment

#Control Derecha
#Avanzar Right
def TwistRight(target_angle):
   
   angular_speed = 0.3  # Velocidad angular
   
   a_in = [0, 0]  # Acción inicial (girar a la derecha)
      # Obtener el estado inicial y el ángulo actual del entorno
   next_state, reward, done, target, current_angle = env.step(a_in)
   kp = -.90
   
   
    # Mientras la diferencia entre el ángulo objetivo y el ángulo actual sea mayor que 1.0
   while True:

        # Si la diferencia sigue siendo mayor que 1.0, continuar girando
        if abs(target_angle - current_angle) > 0.0003:
            # Establecer la velocidad angular en la dirección adecuada
            #a_in[1] = angular_speed if target_angle < current_angle else -(angular_speed-.1)
            # Tomar el siguiente paso en el entorno
            error = target_angle - current_angle
            a_in[1]=kp*error
            next_state, reward, done, target,current_angle = env.step(a_in)
        else:
            # Si la diferencia es menor que 1.0, detener el movimiento
            a_in = [0, 0]
            # Tomar el siguiente paso en el entorno para detener el movimiento
            infoEnvironment = env.step(a_in)
            return infoEnvironment  # Devolver información sobre el entorno
            break  # Salir del bucle while

#Control Izquierd
def TwistLeft(target_angle):
   
   angular_speed = 0.3  # Velocidad angular
   
   a_in = [0, 0]  # Acción inicial (girar a la derecha)
      # Obtener el estado inicial y el ángulo actual del entorno
   next_state, reward, done, target, current_angle = env.step(a_in)
   kp = -.90
   
    # Mientras la diferencia entre el ángulo objetivo y el ángulo actual sea mayor que 1.0
   while True:

        # Si la diferencia sigue siendo mayor que 1.0, continuar girando
        if abs(target_angle - current_angle) > 0.0003:
            # Establecer la velocidad angular en la dirección adecuada
            #a_in[1] = angular_speed if target_angle < current_angle else -(angular_speed-.1)
            # Tomar el siguiente paso en el entorno
            error = target_angle - current_angle
            
            a_in[1]=kp*error
            #print(a_in)
            next_state, reward, done, target,current_angle = env.step(a_in)
        else:
            # Si la diferencia es menor que 1.0, detener el movimiento
            a_in = [0, 0]
            # Tomar el siguiente paso en el entorno para detener el movimiento
            infoEnvironment = env.step(a_in)
            return infoEnvironment  # Devolver información sobre el entorno
            break  # Salir del bucle while




""" action=[0,5]
  a_in = [(action[0] + 1) / 2, action[1]]
  return a_in


#3 grados por vuelta
enviromentInfo = TwistLeft(TurnLeft90)
next_state, reward, done, target,angle = enviromentInfo
print(next_state)
nextStateD=stateDiscrete(next_state)
print(nextStateD)
print(reward)
print(angle)

enviromentInfo = TwistRight(Turn0)
next_state, reward, done, target,angle = enviromentInfo
print(next_state)
nextStateD=stateDiscrete(next_state)
print(nextStateD)
print(reward)
print(angle)
"""


#action_dim=[Forward(),TwistRight(TurnRight45),TwistRight(TurnRight90),TwistLeft(TurnLeft45),TwistLeft(TurnLeft90)]


#Hiperparametros
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20
#Numero de estados y numero de acciones
state_space=11
action_space=5

# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable


Qtable = initialize_q_table(state_space,action_space)

def epsilon_greedy_policy(Qtable, state, epsilon):
  # Randomly generate a number between 0 and 1
  random_int = random.uniform(0,1)
  # if random_int > greater than epsilon --> exploitation
  if random_int > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = np.argmax(Qtable[state])
  # else --> exploration
  else:
    
    
    #action = random.choice(action_dim)
    action = random.randint(0, len(action_dim) - 1)
    
    
  
  return action



def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state])
  
  return action


# Training parameters
n_training_episodes = 100  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters

gamma = 0.95                 # Discounting rate
eval_seed = []  
max_steps =100             # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.0005            # Exponential decay rate for exploration prob



def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in range(n_training_episodes):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()
    stateD=stateDiscrete(state)
    distance=stateD[-2:]
    step = 0
    done = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, stateD, 1)
      
      print("Action",action)
      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done, info, _ = action_dim[action]
      newStateD=stateDiscrete(new_state)
      print(new_stateD)


      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      Qtable[stateD][action] = Qtable[stateD][action] + learning_rate * (reward + gamma * np.max(Qtable[newStateD]) - Qtable[stateD][action]) 

      # If done, finish the episode
      if done: 
        break
      
      # Our state is the new state
      #state = new_state
      stateD=newStateD
      distance=stateD[-2:]
  return Qtable






Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)








"""
enviromentInfo = TwistLeft(TurnLeft90)
next_state, reward, done, target,angle = enviromentInfo
print(next_state)
nextStateD=stateDiscrete(next_state)
print(nextStateD)
print(reward)
print(angle)"""










"""

next_state, reward, done, target = env.step(Forward(),distance[0])
nextStateD=stateDiscrete(next_state)
print(nextStateD)
print(reward)


# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable


Qtable = initialize_q_table(len(stateD),3)


#La navegación será en la parte del ambito local

#Define la politica epsilon gready

def epsilon_greedy_policy(Qtable, state, epsilon):
  # Randomly generate a number between 0 and 1
  random_int = random.uniform(0,1)
  # if random_int > greater than epsilon --> exploitation
  if random_int > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = np.argmax(Qtable[state])
  # else --> exploration
  else:
    action_space = (Forward(),Left(),Right())
    
    action = random.choice(action_space)
  
  return action


def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state])
  
  return action



# Training parameters
n_training_episodes = 100  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters

gamma = 0.95                 # Discounting rate
eval_seed = []  
max_steps =100             # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.0005            # Exponential decay rate for exploration prob



def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in range(n_training_episodes):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()
    stateD=stateDiscrete(state)
    distance=stateD[-2:]
    step = 0
    done = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, stateD, epsilon)
      

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done, info = env.step(action,distance[0])
      newStateD=stateDiscrete(state)
   

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      Qtable[stateD][action] = Qtable[stateD][action] + learning_rate * (reward + gamma * np.max(Qtable[newStateD]) - Qtable[stateD][action]) 

      # If done, finish the episode
      if done:
        break
      
      # Our state is the new state
      state = new_state
      stateD=stateDiscrete(state)
      distance=stateD[-2:]
  return Qtable






Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)

"""

"""
[10.         10.         10.         10.         10.         10.
 10.         10.         10.         10.         10.         10.
 10.         10.         10.         10.         10.         10.
 10.         10.          1.08374886 -1.40797936  0.          0.        ]

#Que son estos estados???

#robot_state = [distance, theta, action[0], action[1]]
#Distancia meta y angulo con respecto al robot 1.08388258 -1.40713085  1.          0.        ]

[ 2.43999521  1.86166022  1.53967781  1.46143949  3.71472969  3.47803828
  3.33035326  3.27595121  3.28054633  3.32216493  3.4650202   3.69499327
  1.67112707  1.44943173  1.31706797  1.22924839  1.18521176  1.16554335
  4.38052598  4.43901252  1.08388258 -1.40713085  1.          0.        ]"""




