
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result, multi_icp


if __name__ == "__main__":
    obj_name = 'drill' # drill or liq_container
    num_pc = 4 # number of point clouds

    source_pc = read_canonical_model(obj_name)

    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)

        # estimated_pose, you need to estimate the pose with ICP
        # Run ICP
        pose = np.eye(4)
        R, t, error = multi_icp(source_pc, target_pc)
        pose[:3, :3] = R
        pose[:3, 3] = t

        # visualize the estimated result
        visualize_icp_result(source_pc, target_pc, pose)