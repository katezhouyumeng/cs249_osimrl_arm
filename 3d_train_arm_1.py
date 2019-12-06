# This file trains the arm's own muscles
# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c


# from osim_rl_master.osim.env.armLocalAct import Arm2DEnv
# from osim_rl_master.osim.env.armLocalAct import Arm2DVecEnv

from osim_rl_master.osim.env.Arm3DEnv import Arm3dEnv

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
env = Arm3dEnv(visualize=True, integrator_accuracy=1e-2)
env.reset()

obs = env.get_observation()
state_dims = obs.shape[0]
state_placeholder = tf.placeholder(tf.float32, [None, state_dims])
# state_action_placeholder = tf.placeholder(tf.float32, [None, state_dims])


def value_function(state):
    n_hidden1 = 64 
    n_hidden2 = 4
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer() # a method of intializing
        # concat = tf.concat([state, action], axis=0)
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.relu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden1, tf.nn.relu, init_xavier)
        hidden3 = tf.layers.dense(hidden2, n_hidden1, tf.nn.relu, init_xavier)
        hidden4 = tf.layers.dense(hidden3, n_hidden1, tf.nn.relu, init_xavier)
        hidden5 = tf.layers.dense(hidden4, n_hidden1, tf.nn.relu, init_xavier)

        # hidden1_action = tf.layers.dense(action, n_hidden1, tf.nn.relu, init_xavier)
        # hidden2 = hidden1 + hidden1_action
        hidden6 = tf.layers.dense(hidden5, n_hidden2, tf.nn.relu, init_xavier) 
        V = tf.layers.dense(hidden6, n_outputs, tf.compat.v1.keras.activations.linear, init_xavier)
    return V

def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40
    # n_hidden3 = 40
    # n_hidden4 = 40
    # n_hidden5 = 10
    n_outputs = 50 # hardcoded number of muscles
    
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
lr_actor = 0.001  
lr_critic = 0.001

# define required placeholders
action_placeholder = tf.placeholder(tf.float32)
# state_action_placeholder = tf.placeholder(tf.float32)
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

# size of a sample
# test_sample = env.observation_space.sample()
# print('size of a sample', )
                                    
state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled                       #returns shape =(1,2)   
###################################################################
###################################################################
# Define fake critic
def fake_critic(state):
    target_x = state[0]
    target_y = state[1]

    pos_x = state[-2]
    pos_y = state[-1]

    penalty = (pos_x - target_x)**2 + (pos_y - target_y)**2
    return -penalty

################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 150
max_step = 70
env.time_limit = max_step
epsilon =0.3
batch_size = 5

# debug code
# env.reset()
# state_desc = env.get_state_desc()
# print(state_desc["joint_pos"]["r_elbow"])
# print(state_desc["markers"])
# act1 = env.action_space.sample()
# env.step(act1)
# print(state_desc)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    for episode in range(num_episodes):
        #receive initial state from E
        goal = env.reset()   # state.shape -> (2,)
        # print(env.get_observation())
        state = np.array(env.get_observation())
        INIT_state = np.array(env.get_observation())

        reward_total = 0 
        steps = 0
        done = False
        # if episode>50:
        #     max_step = 20
        # elif episode>75:
        #     max_step = 50

        while (not done) and steps<max_step:
                
            #Sample action according to current policy
            #action.shape = (1,1)
            # print('what is state', state)
            # just check if policy has been updated
            # print(sess.run(action_tf_var, feed_dict={
            #                   state_placeholder: scale_state(INIT_state)}))

            # test_sample = env.observation_space.sample()
            # print('size of a sample', test_sample.shape)

            # print('state is', state)
            if np.random.random() > epsilon:
                action  = sess.run(action_tf_var, feed_dict={
                              state_placeholder: state.reshape((1,state_dims))})
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
            # next_action = sess.run(action_tf_var, feed_dict={
            #                   state_placeholder: scale_state(next_state)})

            V_of_next_state = sess.run(V, feed_dict = 
                    {state_placeholder: next_state.reshape((1,state_dims))})  

            # V_of_next_state_fake = fake_critic(next_state)
            # print('predicted V', V_of_next_state)
            # print("true V", V_of_next_state_fake)
            #Set TD Target
            #target = r + gamma * V(next_state)     
            target = reward + gamma * np.squeeze(V_of_next_state)

            # td_error = target - V(s)
            #needed to feed delta_placeholder in actor training
            td_error = target - np.squeeze(sess.run(V, feed_dict = 
                        {state_placeholder: state.reshape((1,state_dims))})) 

            # td_error = target - fake_critic(state)
            
            if np.mod(steps, batch_size) ==0:
                #Update critic by minimizinf loss  (Critic training)
                _, loss_critic_val  = sess.run(
                    [training_op_critic, loss_critic], 
                    feed_dict={state_placeholder: state.reshape((1,state_dims)), 
                    target_placeholder: target})
                #Update actor by minimizing loss (Actor training)
                _, loss_actor_val  = sess.run(
                    [training_op_actor, loss_actor], 
                    feed_dict={action_placeholder: np.squeeze(action), 
                    state_placeholder: state.reshape((1,state_dims)), 
                    delta_placeholder: td_error})
            
            state = np.array(next_state)
            #end while
        episode_history.append(reward_total/steps)
        print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            episode, steps, reward_total))
        
        # if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
        #     print("****************Solved***************")
        #     print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(
        #         np.mean(episode_history[-100:])))

# do some ploting
plt.plot(episode_history)
plt.show()


