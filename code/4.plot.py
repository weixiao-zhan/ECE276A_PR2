# %%
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from pr2_utils import plot_odometry

datasets = [20, 21]
odo_strs = ["imu", "icp", "lc0", "lc1"]

for dataset in datasets:
    odo_str = "imu"
    with np.load(f"../data/odometry_{dataset}_{odo_str}.npz") as data:
        imu_odometry = data["X"]
        imu_odometry_stamp = data["stamps"]

    odo_str = "icp"
    with np.load(f"../data/odometry_{dataset}_{odo_str}.npz") as data:
        icp_odometry = data["X"]
        icp_odometry_stamp = data["stamps"]

    odo_str = "lc0"
    with np.load(f"../data/odometry_{dataset}_{odo_str}.npz") as data:
        lc0_odometry = data["X"]
        lc0_odometry_stamp = data["stamps"]

    odo_str = "lc1"
    with np.load(f"../data/odometry_{dataset}_{odo_str}.npz") as data:
        lc1_odometry = data["X"]
        lc1_odometry_stamp = data["stamps"]

    plot_odometry([
        (imu_odometry, imu_odometry_stamp, "imu (motion)", {"linewidth":1}),
        (icp_odometry, icp_odometry_stamp, "icp (observation)",  {"linewidth":1}),
        (lc0_odometry, lc0_odometry_stamp, "factor graph optimized", {"linewidth":1, "linestyle":'--'}),
        (lc1_odometry, lc1_odometry_stamp, "loop closure optimized", {"linewidth":1, "linestyle":'--'}),
    ], {"loc":'lower center', "bbox_to_anchor":(0.5, -0.3), "shadow":False, "ncol":2})


