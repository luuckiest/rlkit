import multiworld
import gym
import mujoco_py
import cv2
from mujoco_py import load_model_from_xml, MjSim, functions
import numpy as np
import rlkit.torch.pytorch_util as ptu
model = mujoco_py.load_model_from_path('/home/xiaomin/Downloads/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_and_reach_mocap_goal_hidden.xml')
sim = MjSim(model)
a = sim.render(width=400, height=400, camera_name='topview', depth=True)
rgb_img = a[0]
modelpos = np.zeros(36)
modelquat = np.zeros(35)
roompos = np.ones(35)
roomquat = np.ones(35)

functions.mjv_room2model(modelpos, modelquat, roompos,
                         roomquat, sim.render_contexts[0].scn)


cv2.imshow('rgb_img', rgb_img)
cv2.waitKey(0)

# multiworld.register_all_envs()
# env = gym.make('SawyerPushNIPS-v0')
# for _ in range(10000):
#     env.render()
