
from osim_rl_master.osim.env.Arm3DEnv import Arm3dEnv
import numpy as np
# from osim_rl_master.osim.env.armLocalAct import Arm2DEnv

# env = Arm3dEnv()
env = Arm3dEnv(visualize=True, integrator_accuracy=1e-1)
env.reset_objective()

# state_desc = env.get_state_desc()

# state_desc.get_
# action = np.array([0]*50)
# action[20] =1


# observation, reward, done, info = env.step(action, obs_as_dict=False)




# print(env.get_state_desc().keys())

if __name__ == '__main__':
    observation = env.reset()
    env.render()

    # observation = env.reset() 

    # obs = env.get_observation()
    # print(obs.shape)

    # sample = env.observation_space.sample()
    # print(env.observation_space.shape[0])
    # print(sample)

    # print('target is')
    # print(env.current_objective)

    # print('observation is')
    # print

    for k in range(1000):

        action = env.action_space.sample()
        action = np.array([0]*50)
        action[20] =1

        observation, reward, done, info = env.step(action, obs_as_dict=False)

        print(observation.shape)
        print('observation is')
        print(observation)

    # action = np.array([0, 0, 0, 0, 1, 0, 0])
    
    # print('shapeline', env.action_space.shape)
    # for i in range(10000):
        # print(i)
        # action = 10*np.random.rand(6,1)
        # observation, reward, done, info = env.step(action, obs_as_dict=False)
        # print(observation)


    # returns a compiled model
    # identical to the previous one
    # model = load_model('/home/lukasz/nnregression.h5')
    # print(model.summary())

    # for i in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         env.reset()