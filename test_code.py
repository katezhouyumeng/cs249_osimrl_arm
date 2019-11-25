from osim_rl_master.osim.env.armLocal import Arm2DEnv

arm = Arm2DEnv(visualize=True)

print("is it local", arm.islocal)