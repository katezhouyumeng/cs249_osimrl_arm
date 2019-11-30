# This file trains the arm's own muscles

from osim_rl_master.osim.env.armLocal import Arm2DEnv
from osim_rl_master.osim.env.armLocal import Arm2DVecEnv

import pprint
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import gym
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import opensim as osim
import tensorflow as tf

from collections import deque

## Initialize environment & set up networks
env = Arm2DVecEnv(visualize=True)

state_dims = env.observation_space.shape[0]
state_placeholder = tf.placeholder(tf.float32, [None, state_dims])


def value_function(state):
    n_hidden1 = 16 
    n_hidden2 = 4
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer() # a method of intializing
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.relu, init_xavier)
        # hidden1_action = tf.layers.dense(action, n_hidden1, tf.nn.relu, init_xavier)
        # hidden2 = hidden1 + hidden1_action
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.relu, init_xavier) 
        V = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
    return V

def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40
    # n_hidden3 = 40
    # n_hidden4 = 40
    # n_hidden5 = 10
    n_outputs = 6 # hardcoded number of muscles
    
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.relu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.relu, init_xavier)
        hidden3 = tf.layers.dense(hidden2, n_hidden2, tf.nn.relu, init_xavier)
        hidden4 = tf.layers.dense(hidden3, n_hidden2, tf.nn.relu, init_xavier)
        hidden5 = tf.layers.dense(hidden4, n_hidden2, tf.nn.relu, init_xavier)
        mu = tf.layers.dense(hidden5, n_outputs, tf.nn.sigmoid, init_xavier)
        sigma = tf.layers.dense(hidden5, n_outputs, tf.nn.sigmoid, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)  # remove the batch dimension

        # hard code action space limits
        action_space_low = 0
        action_space_high = 1
        action_tf_var = tf.clip_by_value(
            action_tf_var, action_space_low, 
            action_space_high) # this returns a tensor flow variable that can be backpropagated
    return action_tf_var, norm_dist


#set learning rates
lr_actor = 0.01  
lr_critic = 0.01

# define required placeholders
action_placeholder = tf.placeholder(tf.float32)
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)

action_tf_var, norm_dist = policy_network(state_placeholder)
V = value_function(state_placeholder)

## DEFINE LOSSES

# define actor (policy) loss function
# loss_actor = -tf.log(K.sum(action_tf_var*action_placeholder) + 1e-5) * delta_placeholder
loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder

training_op_actor = tf.train.AdamOptimizer(
    lr_actor, name='actor_optimizer').minimize(loss_actor)

# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(
                             tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(
        lr_critic, name='critic_optimizer').minimize(loss_critic)


################################################################
#sample from state space for state normalization
import sklearn
import sklearn.preprocessing
                                    
state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled                       #returns shape =(1,2)   
###################################################################


################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 50
max_step = 50
env.time_limit = max_step
epsilon =.3
batch_size = 1



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    for episode in range(num_episodes):
        #receive initial state from E
        goal = env.reset()   # state.shape -> (2,)
        print(env.get_observation())
        state = np.array(env.get_observation())
        INIT_state = np.array(env.get_observation())

        reward_total = 0 
        steps = 0
        done = False
        while (not done) and steps<max_step:
                
            #Sample action according to current policy
            #action.shape = (1,1)
            # print('what is state', state)
            # just check if policy has been updated
            # print(sess.run(action_tf_var, feed_dict={
            #                   state_placeholder: scale_state(INIT_state)}))
            if np.random.random() > epsilon:
                action  = sess.run(action_tf_var, feed_dict={
                              state_placeholder: scale_state(state)})
            else:
                action = np.array(env.action_space.sample())

            action = action.reshape((1, env.action_space.shape[0])) 


            # if np.any(np.isnan(action[0])) or len(action[0]) ==0:
            #     action = env.action_space.sample()

            #Execute action and observe reward & next state from E
            # next_state shape=(2,)    
            #env.step() requires input shape = (1,)
            # print('action taken', action)
            next_state, reward, done, _ = env.step(
                                    np.squeeze(action, axis=0), obs_as_dict=False) 

            steps +=1
            reward_total += reward
            #V_of_next_state.shape=(1,1)
            V_of_next_state = sess.run(V, feed_dict = 
                    {state_placeholder: scale_state(next_state)})  
            #Set TD Target
            #target = r + gamma * V(next_state)     
            target = reward + gamma * np.squeeze(V_of_next_state)


            
            # td_error = target - V(s)
            #needed to feed delta_placeholder in actor training
            td_error = target - np.squeeze(sess.run(V, feed_dict = 
                        {state_placeholder: scale_state(state)})) 
            
            if np.mod(steps, batch_size) ==0:
                #Update actor by minimizing loss (Actor training)
                _, loss_actor_val  = sess.run(
                    [training_op_actor, loss_actor], 
                    feed_dict={action_placeholder: np.squeeze(action), 
                    state_placeholder: scale_state(state), 
                    delta_placeholder: td_error})
                #Update critic by minimizinf loss  (Critic training)
                _, loss_critic_val  = sess.run(
                    [training_op_critic, loss_critic], 
                    feed_dict={state_placeholder: scale_state(state), 
                    target_placeholder: target})
            
            state = np.array(next_state)
            #end while
        episode_history.append(reward_total/steps)
        print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            episode, steps, reward_total))
        
        if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
            print("****************Solved***************")
            print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(
                np.mean(episode_history[-100:])))

# do some ploting
plt.plot(episode_history)
plt.show()


