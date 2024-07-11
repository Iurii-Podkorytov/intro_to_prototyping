import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import mujoco
import mujoco.viewer

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('test.xml') 
data = mujoco.MjData(model)

# Define the range of motor angles (in radians)
angle_ranges = (-np.pi, np.pi)

# Define the number of steps for each joint
steps = 10

# Generate angle arrays for each joint
angles_1 = np.linspace(*angle_ranges, steps)
angles_2 = np.linspace(*angle_ranges, steps)
angles_3 = np.linspace(*angle_ranges, steps)

# Create lists to store the results
joint_angles = []
torques = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Iterate through all possible configurations
    for i in angles_1:
        for j in angles_2:
            for k in angles_3:
                # Set joint angles
                data.qpos = np.array([i, j, k])

                # Calculate the inverse dynamics
                mujoco.mj_inverse(model, data)

                # Save the joint angles and torques
                joint_angles.append([i, j, k])
                torques.append(data.qfrc_inverse)

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

# Convert the lists to NumPy arrays
joint_angles = np.array(joint_angles)
torques = np.array(torques)

# Create a violin plot of the torques
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.violinplot(torques[:, i], positions=[i], showmeans=True)

plt.xticks(range(3), ("hinge_1", "hinge_2", "hinge_3"))
plt.xlabel('Joint Name')
plt.ylabel('Torque')
plt.title('Distribution of Torques in Different Joints')
plt.show()