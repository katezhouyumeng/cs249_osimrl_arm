
from osim_rl_master.osim.env.armLocalAct import Arm2DEnv
from osim_rl_master.osim.env.armLocalAct import Arm2DVecEnv
import numpy as np
import matplotlib.pyplot as plt


# create the arm
env = Arm2DEnv(visualize=True)
env.render()


pos_all_x = []
pos_all_y = []
pos_all_z = []

if __name__ == '__main__':
    observation = env.reset() 
    # which actions do more work, 1 and 4
    action = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    # action = env.action_space.sample()
    for i in range(500):
        print(i)
    	# action = 10*np.random.rand(6,1)
        observation, reward, done, info = env.step(action, obs_as_dict=True)
        # observation, reward, done, info = env.step(action)
        humerus_pos = observation["markers"]["r_radius_styloid"]["pos"]
        pos_all_x.append(humerus_pos[-2])
        pos_all_y.append(humerus_pos[-1])

    	# # humerus_pos = observation['body_pos']['r_ulna_radius_hand']
    	# pos_all_x.append(humerus_pos[0])
    	# pos_all_y.append(humerus_pos[1])
    	# pos_all_z.append(humerus_pos[2])
# 
# print('position all x', pos_all_x)
# print('position all y', pos_all_y)
# print('position all z', pos_all_z)

# print(env.observation_space.high)

# ploting traj
fig = plt.figure()

ax = fig.add_subplot(211)
ax.plot(pos_all_x)
ax.title.set_text('x')
ax = fig.add_subplot(212)
ax.plot(pos_all_y)
ax.title.set_text('y')
# ax = fig.add_subplot(313)
# ax.plot(pos_all_z)
# ax.title.set_text('z')
plt.show()
# plt.plot(pos_all_x)