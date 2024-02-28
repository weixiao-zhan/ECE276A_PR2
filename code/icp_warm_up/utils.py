import os
import scipy.io as sio
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def read_canonical_model(model_name):
    '''
    Read canonical model from .mat file
    model_name: str, 'drill' or 'liq_container'
    return: numpy array, (N, 3)
    '''
    model_fname = os.path.join('./data', model_name, 'model.mat')
    model = sio.loadmat(model_fname)

    cano_pc = model['Mdata'].T / 1000.0 # convert to meter

    return cano_pc


def load_pc(model_name, id):
    '''
    Load point cloud from .npy file
    model_name: str, 'drill' or 'liq_container'
    id: int, point cloud id
    return: numpy array, (N, 3)
    '''
    pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
    pc = np.load(pc_fname)

    return pc


def visualize_icp_result(source_pc, target_pc, pose):
    '''
    Visualize the result of ICP
    source_pc: numpy array, (N, 3)
    target_pc: numpy array, (N, 3)
    pose: SE(4) numpy array, (4, 4)
    '''
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
    source_pcd.paint_uniform_color([0, 0, 1])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
    target_pcd.paint_uniform_color([1, 0, 0])

    source_pcd.transform(pose)

    o3d.visualization.draw_geometries([source_pcd, target_pcd])


def find_closest_points(source, target):
    """
    Find the closest points in the target for each point in the source using Scikit-learn's NearestNeighbors.
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(target)
    _, indices = neigh.kneighbors(source, return_distance=True)
    closest_points = target[indices.flatten()]
    return closest_points

def estimate_transformation(source, target):
    """
    Estimate the rotation and translation using Singular Value Decomposition (SVD).
    """
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    Q =  np.dot((target - target_centroid).T, (source - source_centroid))
    U, _, Vt = np.linalg.svd(Q)
    R = (U @ Vt)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = (U @ Vt)
    t = target_centroid - R @ source_centroid
    return R, t

def apply_transformation(source, R, t):
    """
    Apply the estimated rotation and translation to the source points.
    """
    return (R @ source.T).T + t.squeeze()

def icp(source, target,
        R = np.eye(3), t = np.zeros(3),
        iterations=80, tolerance=1e-8):
    '''
    return target_T_source
    '''
    source_t = apply_transformation(source, R, t)

    prev_error = float('inf')
    for i in range(iterations):
        closest_points = find_closest_points(source_t, target)
        R, t = estimate_transformation(source_t, closest_points)
        source_t = apply_transformation(source_t, R, t)
        error = np.mean(np.sum((closest_points - source_t) ** 2, axis=1))
        # bar.set_postfix_str(f"error: {error}")
        if np.abs(prev_error - error) < tolerance:
            break
        prev_error = error
    R, t = estimate_transformation(source, closest_points)
    return R, t, error

def multi_icp(source, target, trial = 12):
    best_R, best_t, best_error = np.eye(3), np.zeros(3), float("inf")
    for yaw in np.linspace(0, 2*np.pi, trial):
        rr = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0,0,1]
        ])
        tt = np.array([1,0,0])
        R, t, error = icp(source, target, rr, tt)
        if error < best_error:
            best_R = R
            best_t = t
    return best_R, best_t, best_error

def o3d_icp(source_PC, target_PC, trans_init = np.eye(4), threshold = 5):
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
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # Extract the translation from the transformation matrix
    return reg_p2p.transformation

def o3d_fpfh(source_PC, target_PC, trans_init=np.eye(4)):
    # Voxel downsampling for both point clouds
    voxel_size = 0.1  # You can adjust this value
    source_down = source_PC.voxel_down_sample(voxel_size)
    target_down = target_PC.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    # Initial transformation
    result.transformation = np.dot(result.transformation, trans_init)
    
    # You may want to refine the alignment further using ICP
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, distance_threshold, result.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result.transformation  # or result_icp.transformation for ICP refinement
