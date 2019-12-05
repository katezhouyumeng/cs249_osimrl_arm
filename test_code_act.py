from osim_rl_master.osim.env.armLocalAct import Arm2DVecEnv
from osim_rl_master.osim.env.armLocalAct import Arm2DEnv



arm = Arm2DEnv(visualize=True)

print("is it local", arm.islocal)
arm.reset()
arm.render()
print(arm.get_observation())
print(arm.action_space.shape[0])

for k in range(500):
	# action = arm.action_space.sample()
	# print(action)
	if k <10:
		action = [0,0,0,0,0,0,10]
	else:
		action = [0,0,0,0,0,0,10]
	print(action)
	new_state, reward, done, _ = arm.step(action, obs_as_dict=False)
	print(arm.istep)

