
from osim_rl_master.osim.env.Arm3DEnv import Arm3dEnv
# from osim_rl_master.osim.env.armLocalAct import Arm2DEnv

# env = Arm3dEnv()
env = Arm3dEnv(visualize=True, integrator_accuracy=1e-4)

if __name__ == '__main__':
    observation = env.reset()
    env.render()

    observation = env.reset() 
    # action = np.array([0, 0, 0, 0, 1, 0, 0])
    action = env.action_space.sample()
    print('shapeline', env.action_space.shape)
    for i in range(10000):
        print(i)
        # action = 10*np.random.rand(6,1)
        observation, reward, done, info = env.step(action, obs_as_dict=True)

    # returns a compiled model
    # identical to the previous one
    # model = load_model('/home/lukasz/nnregression.h5')
    # print(model.summary())

    # for i in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         env.reset()