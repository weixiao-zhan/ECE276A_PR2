# %%
from icp_warm_up.utils import icp, o3d_icp, icp_partial
from pr2_utils import *

# %%
dataset = 21
with np.load(f"../data/Hokuyo{dataset}.npz") as data:
    lidar_ranges = data["ranges"].T       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    print(data["angle_increment"])
with np.load(f"../data/odometry_{dataset}_imu.npz") as data:
    imu_odometry = data["X"]
    imu_odometry_stamp = data["stamps"]

f_imu_odometry = InterpN1D(imu_odometry_stamp, imu_odometry)

# %%
plot_scan(lidar_ranges[1400], f_imu_odometry(lidar_stamps[1400]))
plot_scan(lidar_ranges[1401], f_imu_odometry(lidar_stamps[1401]))

icp_partial(
    lidar_scan_to_3dpc(lidar_ranges[1401,:]),
    lidar_scan_to_3dpc(lidar_ranges[1400,:]),
)

# %%
n = lidar_stamps.shape[0]
icp_T = np.zeros([n,4,4])
icp_T[0] = np.eye(4)
imu_T = odometry_to_transformation(f_imu_odometry(lidar_stamps))
sum_error = 0
for i in tqdm(range(1, n)):
    T_guess = diff_transformation(imu_T[i], imu_T[i-1])
    T, error = icp_partial(
        lidar_scan_to_3dpc(lidar_ranges[i,:]),
        lidar_scan_to_3dpc(lidar_ranges[i-1,:]),
        T_guess
    )
    sum_error += error
    icp_T[i] = icp_T[i-1] @ T
print(sum_error / n)
icp_odometry = transformation_to_odometry(icp_T)

# %%
# with np.load(f"../data/odometry_icp_{dataset}.npz") as data:
#     odometry_scan_read = data["X"]

# %%
plot_odometry([
    (imu_odometry,imu_odometry_stamp, "imu (motion)"),
    (icp_odometry, lidar_stamps, "icp (observation)"),
])

# %%
np.savez(f'../data/odometry_{dataset}_icp.npz', X=icp_odometry, stamps=lidar_stamps)


