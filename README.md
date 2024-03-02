# main code
[`code/1.odometry.py`](code/1.odometry.py): this script will use motion model (differential-drive kinematics) to build trajectory and save to `data/odometry_{dataset}_imu.npz`

[`code/2.scan_matching.py`](code/2.scan_matching.py): this script will use observation model (ICP) to build trajectory and save to `data/odometry_{dataset}_icp.npz`

[`code/3.loop_closure.py`](code/3.loop_closure.py): this script will read in `data/odometry_{dataset}_imu.npz` and `data/odometry_{dataset}_icp.npz`, perform factor graph loop closure.
Outputs are :
1. `data/odometry_{dataset}_lc0.npz`: the trajectory from factor graph optimization **without** loop closure constrains.
2. `data/odometry_{dataset}_lc1.npz`: the trajectory from factor graph optimization **with** loop closure constrains.

[`code/4.plot.py`](code/4.plot.py): this script will plot all the trajectory.

[`code/5.mapping.py`](code/5.mapping.py): this script will perform occupancy map and texture map on given dataset and trajectory.

*You are welcome to use the jupyter nootbook (\*.ipynb) instad of listed python script (\*.py) .*

# helper code
[`code/icp_warm_up/utils.py`](code/icp_warm_up/utils.py): my ICP and proportion ICP implementation

[`code/pr2_utils.py`](code/pr2_utils.py): common helper class and functions used by main code.

# results
main dataset: [`img/`](/img/): 

icp warm up: [`code/icp_warm_up/`](code/icp_warm_up/)
