import gym

import random
import numpy as np
from collections import deque
print("Gym:", gym.__version__)
import matplotlib.pyplot as plt
from unn import nn_arch,relu,relu_backward,single_layer_forward_propagation,full_forward_propagation,u_full_backward_propagation,single_layer_backward_propagation
from vnn import update,init_layers
from Adam import initialize_adam,update_parameters_with_adam
from baseline import update_b,b_full_backward_propagation
env_name = "Pendulum-v0"
env = gym.make(env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
#Hyperparameters
NUM_EPISODES = 5
learning_rate = 0.001
GAMMA = 0.99
nn_architecture=nn_arch()
u_params_values=init_layers(nn_architecture,seed=99)
v_params_values=init_layers(nn_architecture,seed=99)
state=env.reset()
state=state.reshape(3,1)
#baseline
b_params_values=init_layers(nn_architecture,seed=99)
env.reset()
_,reward,_,_ = env.step([2])
b,c3=full_forward_propagation(state,b_params_values,nn_architecture)
b_grads=b_full_backward_propagation(b,reward,c3, b_params_values, nn_architecture)
b_delta_sum={}
for key in b_grads:
    b_delta_sum[key]=np.zeros((b_grads[key].shape))
def policy(state,w1,w2):
    
    u,c1 = full_forward_propagation(state, u_params_values, nn_architecture)

    
    v,c2= full_forward_propagation(state,v_params_values,nn_architecture)
    
    
    a=np.random.normal(np.squeeze(u),np.squeeze(v))
    #print(a)
    return a,v,u,c1,c2
def gradients(a,v,u,u_params_values,v_params_values,nn_architecture,c1,c2):
    
    v_grads= v_full_backward_propagation(v, a,u, c2, v_params_values, nn_architecture)
    u_grads= u_full_backward_propagation(u, a, v,c1, u_params_values, nn_architecture)
    #print(v_grads)
    #print(u_grads)
    return v_grads,u_grads
a,v,u,c1,c2=policy(state,u_params_values,v_params_values)
u_grads= u_full_backward_propagation(u, a, v,c1, u_params_values, nn_architecture)
delta_sum={}
for key in u_grads:
    delta_sum[key]=np.zeros((u_grads[key].shape))
    
#Hyperparameters
NUM_EPISODES = 10
learning_rate = 0.01
GAMMA = 0.99
#weight update interval
weight_update_interval=10
weights_dict={}
for i in u_params_values:
    weights_dict[i]=np.zeros((u_params_values[i].shape))
u_params_values_avg=[]    
#u_params_values_avg=0
#v_params_values_avg=0
# Keep stats for final print of graph
episode_rewards = []
num_eps=[]
#neural network architecture
nn_architecture=nn_arch()
#initailse
u_params_values=init_layers(nn_architecture,seed=99)
b_params_values=init_layers(nn_architecture,seed=99)
#v_params_values=init_layers(nn_architecture,seed=99)
# Our policy that maps state to action parameterized by w1,w2

def policy(state,w1):
    
    u,c1 = full_forward_propagation(state, w1, nn_architecture)

    
    #v,c2= full_forward_propagation(state,v_params_values,nn_architecture)
    v=0.0009
    
    a=np.random.normal(np.squeeze(u),(v))
    #print(a)
    a=np.clip(a,-2,+2)
    return a,v,u,c1

u_delta_sum={}
#v_delta_sum={}
#for key in v_grads:
    #v_delta_sum[key]=np.zeros((v_grads[key].shape))
for key in u_grads:
    u_delta_sum[key]=np.zeros((u_grads[key].shape))
    
# Main loop 

for e in range(1,NUM_EPISODES+1):

    state = env.reset()
    b_grads1=[]
    grads1 = []
    grads2=[]
    rewards = []
    u_delta_sum={}
    b_delta_sum={}
    for key in u_grads:
        u_delta_sum[key]=np.zeros((u_grads[key].shape))
    for key in b_grads:
        b_delta_sum[key]=np.zeros((b_grads[key].shape))
    # Keep track of game score to print
    score = 0
    #v,s=initialize_adam(u_params_values)
    t=0
    while True:
        
        state=state.reshape(3,1)
        state=state/env.observation_space.high
        a,v,u,c1=policy(state,u_params_values)
        b,c3=full_forward_propagation(state,b_params_values,nn_architecture)
        next_state,reward,done,_ = env.step(a)
        # Compute gradient 
        
        u_grads= u_full_backward_propagation(u, a, v,c1, u_params_values, nn_architecture)
        #u_params_values,v,s=update_parameters_with_adam(u_params_values, u_grads, v, s, t, learning_rate ,
              #                  beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8)
        b_grads=b_full_backward_propagation(b,reward,c3, b_params_values, nn_architecture)
        grads1.append(u_grads)
        b_grads1.append(b_grads)
        #grads2.append(v_grads)
        rewards.append(reward-b)
        
        score+=reward
        state = next_state
        t=t+1
        if done:
             break
    
    for i in grads1:
        for j in u_grads:
            u_delta_sum[j]+=i[j]
    for i in b_grads1:
        for j in b_grads:
            b_delta_sum[j]+=i[j]
    
    b_params_values=update_b(b_params_values, b_delta_sum, nn_architecture, learning_rate)
            
    #print(u_delta_sum)
    u_params_values_avg.append(update(u_params_values, nn_architecture, learning_rate,rewards,GAMMA,u_delta_sum))
    if(e%weight_update_interval==0):
        for i in u_params_values_avg:
            for j in u_params_values:
                weights_dict[j]+=i[j]
                
        for i in weights_dict:
            weights_dict[i]=weights_dict[i]/weight_update_interval
            u_params_values[i]=weights_dict[i]
            
        for i in weights_dict:
            weights_dict[i]=np.zeros((u_params_values[i].shape))
        u_params_values_avg=[]
                
        
    #v_params_values=update(v_params_values, nn_architecture, learning_rate,rewards,GAMMA,v_delta_sum)
    #if(e%weight_update_interval)
    episode_rewards.append(score)  #rewards 
    num_eps.append(e) #number of episodes
    
num_eps=np.squeeze(num_eps)
num_eps=num_eps[:,None]
episode_rewards=np.squeeze(episode_rewards)
episode_rewards=episode_rewards[:,None]

#plotting rewards vs episodes
plt.xlabel('number of episodes')
plt.ylabel('Total reward')
plt.plot(num_eps,episode_rewards)