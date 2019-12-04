"""
solving pendulum using actor-critic model
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
"""
from osim_rl_master.osim.env.armLocal import Arm2DEnv
from osim_rl_master.osim.env.armLocal import Arm2DVecEnv

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten,concatenate
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
import opensim as osim
import sys
import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.01
		self.epsilon = 0.3
		self.epsilon_decay = 1   # how much exploration is decaying - not decaying
		self.gamma = .99
		self.tau   = 0.001

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=2000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		# separate models
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)
		
		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		h1 = Dense(32, activation='relu')(state_input)
		h2 = Dense(32, activation='relu')(h1)
		h3 = Dense(32, activation='relu')(h2)
		output = Dense(self.env.action_space.shape[0], activation='sigmoid')(h3)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		print(model.summary())

		return state_input, model

	def create_critic_model(self):
		# print("what is shape",self.env.observation_space.shape)
		# state_input = Input(shape=self.env.observation_space.shape[0])
		state_input = Input(shape=self.env.observation_space.shape)
		# flattened_observation = Flatten()(observation_input)

		action_input = Input(shape=self.env.action_space.shape)
		# flattened_action = Flatten()(action_input)

		merged    = concatenate([state_input, action_input])


		state_h1 = Dense(64, activation='relu')(merged)
		state_h2 = Dense(64, activation='relu')(state_h1)
		state_h3 = Dense(64, activation='relu')(state_h2)
		
		# action_h1    = Dense(64)(action_input)
		
		# merged_h1 = Dense(2, activation='relu')(merged)
		output = Dense(1, activation='linear')(state_h3)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		print(model.summary())
		return state_input, action_input, model

			###################################################################
	# Define fake critic
	def fake_critic(self,state):
	    target_x = state[0]
	    target_y = state[1]

	    pos_x = state[-2]
	    pos_y = state[-1]

	    penalty = (pos_x - target_x)**2 + (pos_y - target_y)**2

	    reward =  1.-penalty
	    sd = self.env.get_state_desc()
	    wr_x = sd["markers"]["r_humerus_epicondyle"]["pos"][0]

	    if wr_x > pos_x:
	    	reward -=50

	    return reward

################################################################

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
		# reward_all = 0
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward_fake = self.fake_critic(new_state[0])
				future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				reward += self.gamma * future_reward_fake
			# print (cur_state)
			# print(action)
			# print(reward)
			# print(target_action)
			self.critic_model.fit([cur_state, action], [reward], verbose=0)
			# reward_all+= reward
		# return reward_all
		
	def train(self, batch_size):
		# batch_size = 32
		# if len(self.memory) < batch_size:
		# 	return
		# rewards = []
		# samples = random.sample(self.memory, batch_size) # not sure why randomly sample
		# samples = self.memory.pop()
		samples = [self.memory.popleft() for _i in range(batch_size)]
		# if batch_size == 1:
		# 	samples = [samples]
		# print('SAMPLES ARE', samples)
		reward_all = 0
		for s in samples:
			cur_state, action, reward, new_state, done = s
			reward_all += reward

		# order of these swapped
		self._train_critic(samples)
		self._train_actor(samples)
		return reward_all

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = self.tau*actor_model_weights[i] + (1-self.tau)*actor_target_weights[i] 
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = self.tau*critic_model_weights[i] + (1-self.tau)*critic_target_weights[i]
		self.target_critic_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_critic_target()
		self._update_actor_target()
		

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return self.actor_model.predict(cur_state)

def main():
	sess = tf.Session()
	K.set_session(sess)  # not sure about this, probably just setting to keras
	env = Arm2DVecEnv(visualize=True)

	actor_critic = ActorCritic(env, sess)

	num_episode = 500  # number of episodes
	episode_len  = 200
	batch_size = 32

	env.time_limit = episode_len

	# keep track of reward per episode
	episode_reward =[0]*num_episode
	episode_steps =[0]*num_episode

	for e in range(num_episode):
		# intializations for each episode
		k = 0
		done = 0 
		cur_target = env.reset()
		cur_state = np.array(env.get_observation())
		print('INIT state', cur_state)
		# action = env.action_space.sample()
		# clear th ememory queue
		# actor_critic.memory = deque(maxlen=2000)
		r_all = 0

		# if e>50:
		# 	episode_len = 20
		# elif e>75:
		# 	episode_len = 50

		while not done and k<episode_len:
			cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
			# print('?@?$ current state is', cur_state)
			action = actor_critic.act(cur_state)
			action = action.reshape((1, env.action_space.shape[0])) 
			# print('action is', action)
			action[0] = np.clip(action[0], 0 ,1)
			# print('action after clipping is', action)

			if np.any(np.isnan(action[0])) or len(action[0]) ==0:
				continue
				action = actor_critic.env.action_space.sample()

			# print('what is the action', action[0])
			new_state, reward, done, _ = env.step(action[0], obs_as_dict=False)
			# env.integrate()

			# if reward<0.99:
			# 	done = False
			# else:
			# 	done = True
				# something wrong with their done function, always end at 201
			# if k == 199 or  k==26:
			# 	done = False
			# 	print('taking action', action[0])
			# # print("is it done", done)
			# if done:
			# 	# env.render()
			# 	# reward2 = env.reward()
			# 	# print("reward 1", reward)
			# 	# print("reward 2", reward2)
			# 	# print('istep', env.istep)
			# 	# print("Sample a few times")
			# 	# print(env.action_space.sample())
			# 	# print(env.action_space.sample())
			# 	# print(env.action_space.sample())
			# 	# print(env.action_space.sample())
			# 	# print(env.action_space.sample())

			# 	input("Press Enter to continue...")


			# print("new state is", new_state)
			new_state = np.array(new_state)
			new_state = new_state.reshape((1, env.observation_space.shape[0]))

			actor_critic.remember(cur_state, action, reward, new_state, done)
			if len(actor_critic.memory) > batch_size:
				# sample = random.sample(actor_critic.memory, 1)
				r_all = actor_critic.train(batch_size)
				episode_reward[e] += r_all
			actor_critic.update_target()

			cur_state = new_state
			k += 1

			print("\rEpisode {} @ Time_Step {}/{} ({})".format(
	                    e, k+1, episode_len, episode_reward[e]/k), end="")
			# print(done)

		episode_steps[e] = k
		episode_reward[e] = episode_reward[e]/k

	return episode_reward, episode_steps

if __name__ == "__main__":
	episode_reward, episode_steps = main()

	plt.plot(episode_reward)
	plt.ylabel('Total Reward')
	plt.xlabel('Episode Number')
	plt.show()

	plt.plot(episode_steps)
	plt.ylabel('Total Steps to Reach Goal')
	plt.xlabel('Episodes Number')
	plt.show()