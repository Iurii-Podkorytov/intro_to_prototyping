import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import mujoco
import mujoco.viewer

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('hw2\\task2\\3_joint_manipulator_with_motors.xml') 
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:


    mujoco.mj_step(model, data)
    viewer.sync()

