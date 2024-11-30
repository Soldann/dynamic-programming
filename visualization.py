import sys

import matplotlib.pyplot as plt
import numpy as np
from Constants import Constants
from utils import *

try:
    J_opt = np.load("workspaces/J_opt.npy")
    u_opt = np.load("workspaces/u_opt.npy")
except FileNotFoundError:
    print("Please run the main.py script first to compute the optimal cost and policy.")
    sys.exit()

M = Constants.M
N = Constants.N
START_POS = Constants.START_POS
GOAL_POS = Constants.GOAL_POS
DRONE_POS = Constants.DRONE_POS
STATE_SPACE = Constants.STATE_SPACE
INPUT_SPACE = Constants.INPUT_SPACE

swan_x = None
swan_y = None

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Click on the grid to place the swan", fontsize=14)


def setup_axes(ax):
    ax.set_xlabel("X-axis (East-West)")
    ax.set_ylabel("Y-axis (South-North)")
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(M), minor=False)
    ax.set_yticks(range(N), minor=False)
    ax.set_xticks(np.arange(-0.5, M - 0.5 + 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N - 0.5 + 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.grid(which="major", linestyle="", linewidth=0)
    ax.set_axisbelow(True)

setup_axes(ax)

ax.scatter(START_POS[0], START_POS[1], color="green", s=100, label="Start", zorder=3)
ax.scatter(GOAL_POS[0], GOAL_POS[1], color="red", s=100, label="Goal", zorder=3)
ax.scatter(
    DRONE_POS[:, 0],
    DRONE_POS[:, 1],
    color="orange",
    s=100,
    label="Static Drones",
    zorder=3,
)

(swan_marker,) = ax.plot(
    [], [], "o", color="purple", markersize=10, label="Swan", zorder=3
)

cbar = None


def update_plots(swan_x, swan_y):
    global cbar  # Declare cbar as global to modify it inside the function

    swan_positions = (STATE_SPACE[:, 2] == swan_x) & (STATE_SPACE[:, 3] == swan_y)

    if not np.any(swan_positions):
        print(f"No states found for swan at position ({swan_x}, {swan_y}).")
        return

    robot_positions = STATE_SPACE[swan_positions][:, :2]
    robot_indices = np.where(swan_positions)[0]

    J_opt_grid = np.full((N, M), np.nan)  # Use np.nan to mask positions
    u_opt_grid = np.full((N, M, 2), np.nan)

    # Fill in the cost-to-go and optimal policy for each robot position
    for idx, pos in zip(robot_indices, robot_positions):
        x, y = pos  # x and y positions of the robot
        if 0 <= x < M and 0 <= y < N:
            J_opt_grid[y, x] = J_opt[idx]
            u_opt_grid[y, x, :] = Constants.INPUT_SPACE[u_opt[idx]]

    # Mask the static drone positions
    for drone_pos in DRONE_POS:
        x, y = drone_pos
        if 0 <= x < M and 0 <= y < N:
            J_opt_grid[y, x] = np.nan  # Mask the cost-to-go value
            u_opt_grid[y, x, :] = np.nan  # Mask the control input

    # Remove existing colorbar if any
    if cbar is not None:
        cbar.remove()

    # Clear previous plots
    ax.cla()
    # Update plot settings
    ax.set_title(f"Policy with Swan at ({swan_x}, {swan_y}).", fontsize=14)
    setup_axes(ax)

    # Plot the cost-to-go heatmap, using a masked array to handle np.nan
    im = ax.imshow(
        J_opt_grid,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(-0.5, M - 0.5, -0.5, N - 0.5),
        zorder=1,
    )

    # Create color bar
    cbar = fig.colorbar(im, ax=ax)

    # Plot start, goal, drones, and swan positions
    ax.scatter(
        START_POS[0], START_POS[1], color="green", s=100, label="Start", zorder=3
    )
    ax.scatter(GOAL_POS[0], GOAL_POS[1], color="red", s=100, label="Goal", zorder=3)
    ax.scatter(
        DRONE_POS[:, 0],
        DRONE_POS[:, 1],
        color="orange",
        s=100,
        label="Static Drones",
        zorder=3,
    )
    ax.scatter(swan_x, swan_y, color="purple", s=100, label="Swan", zorder=3)

    X, Y = np.meshgrid(np.arange(M), np.arange(N))

    u_x = u_opt_grid[:, :, 0]
    u_y = u_opt_grid[:, :, 1]

    # Mask the static drone positions in the action vectors
    u_x_masked = np.ma.array(u_x, mask=np.isnan(u_x))
    u_y_masked = np.ma.array(u_y, mask=np.isnan(u_y))

    # Plot the vector field (arrows based on the flow field)
    ax.quiver(
        X,
        Y,
        Constants.FLOW_FIELD[:, :, 0],
        Constants.FLOW_FIELD[:, :, 1],
        angles="xy",
        scale_units="xy",
        color="red",
        width=0.003,
        headwidth=3,
        headlength=4,
        zorder=3,
        label="Flow Field",
    )

    # Plot the vector field (arrows based on the input)
    ax.quiver(
        X,
        Y,
        u_x_masked,
        u_y_masked,
        angles="xy",
        scale_units="xy",
        color="blue",
        width=0.003,
        headwidth=3,
        headlength=4,
        zorder=2,
        label="Input",
    )

    # Remove legend and readd
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(loc="upper left", framealpha=0.5)

    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes != ax:
        return

    x = int(round(event.xdata))
    y = int(round(event.ydata))

    if event.button == 1:  # Left mouse button
        if 0 <= x < M and 0 <= y < N:
            global swan_x, swan_y
            swan_x, swan_y = x, y
            print(f"Swan placed at position ({swan_x}, {swan_y})")
            swan_marker.set_data([swan_x], [swan_y])
            update_plots(swan_x, swan_y)


cid = fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()