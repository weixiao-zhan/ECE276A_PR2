import numpy as np
import open3d as o3d

def icp_registration(source_PC, target_PC, trans_init = np.eye(4), threshold = 0.00002):
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_PC)
    target.points = o3d.utility.Vector3dVector(target_PC)
    
    # Apply ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    # Extract the translation from the transformation matrix
    return reg_p2p.transformation