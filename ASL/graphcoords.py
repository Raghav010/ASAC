import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm

# Read the CSV file
filename = 'new_out.csv'

with open(filename, 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Extract x, y, and depth (z) from each row
x_values = [(2*float(rows[1][i])-1) for i in range(1, len(rows[1]), 3)]  # Every 1st value
y_values = [(2*(1-float(rows[1][i]))-1) for i in range(2, len(rows[1]), 3)]  # Every 2nd value
depth_values = [(1-float(rows[1][i])) for i in range(3, len(rows[1]), 3)]  # Every 3rd value



# Calculate the color gradient based on frame count (index i)
frame_count = len(x_values)
colors = np.linspace(0, 1, frame_count)  # Linear gradient from 0 (blue) to 1 (red)
colors = cm.Reds(colors)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D space
sc = ax.scatter(x_values, y_values, depth_values, c=colors)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')
ax.set_title('Trajectory')

# Set the range of x, y, and depth axes to -1 to 1
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Invert the y-axis
ax.invert_yaxis()

# Add color bar
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Time')

plt.show()

