import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
from scipy.interpolate import interp1d
from tqdm import tqdm

def tic():
    return time.time()
def toc(tstart, name="Operation"):
   print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

##### 1. 
class InterpN1D:
    def __init__(self, t, values, kind='linear'):
        """
        Initialize the interpolation object.

        :param t: The timestamps for the data points.
        :param values: A 2D array of shape (n_samples, n_series) containing the data points.
        :param kind: The type of interpolation (e.g., 'linear', 'cubic'). Defaults to 'linear'.
        """
        self.fs = [interp1d(t, values[:, i], kind=kind, fill_value='extrapolate') for i in range(values.shape[1])]
    
    def __call__(self, ts):
        """
        Interpolate the values at the new timestamps.

        :param ts: An array of new timestamps to interpolate the values at.
        :return: A 2D array of the interpolated values.
        """
        # Interpolate each series and stack them horizontally
        return np.array([f(ts) for f in self.fs]).T

def plot_odometry(odometries, loc = "best"):
    '''
    odometries: [(odometry1, timestamp1, label), (odometry2, timestamp2, label), ...]
    '''
    for odometry, stamp, label in odometries:
        plt.plot(odometry[:,0], odometry[:,1], label=label)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.legend(loc=loc)
    plt.grid(True)
    plt.show()

    for i, (odometry, stamp, label) in enumerate(odometries):
        plt.plot(stamp, odometry[:,0], label=label)
    plt.xlabel("timestamp")
    plt.ylabel("x")
    plt.legend(loc=loc)
    plt.show()

    for i, (odometry, stamp, label) in enumerate(odometries):
        plt.plot(stamp, odometry[:,1], label=label)
    plt.xlabel("timestamp")
    plt.ylabel("y")
    plt.legend(loc=loc)
    plt.show()

    for i, (odometry, stamp, label) in enumerate(odometries):
        plt.plot(stamp, odometry[:,2], label=label)
        if np.min(odometry[:,2])<-np.pi or np.max(odometry[:,2])>np.pi:
            normalized_yaw = np.remainder(odometry[:,2]+np.pi, 2*np.pi)-np.pi
            plt.plot(stamp, normalized_yaw, color = f'C{i}', linestyle=":",label=f'{label} [normalized]')
    plt.xlabel("timestamp")
    plt.ylabel("yaw")
    plt.legend(loc=loc)
    plt.show()

##### 2.
lidar_angle_min, lidar_angle_max, lidar_range_min, lidar_range_max = -2.35619449, 2.35619449, 0.1, 30
lidar_num_points = 1081
lidar_angles = np.linspace(lidar_angle_min, lidar_angle_max, lidar_num_points)

def plot_scan(lidar_ranges, odometry):
    pc = lidar_scan_to_3dpc(lidar_ranges)
    wTo = odometry_to_transformation(odometry)
    r, t = get_Rt(wTo)
    pc_world = (r @ pc.T).T + t
    plt.scatter(pc_world[:,0], pc_world[:,1])

    dx = 3 * np.cos(odometry[2])
    dy = 3 * np.sin(odometry[2])
    plt.arrow(odometry[0], odometry[1], dx, dy, head_width=0.1, head_length=0.15, fc='red', ec='red')
    
    dx = 3 * np.cos(odometry[2]+lidar_angle_max)
    dy = 3 * np.sin(odometry[2]+lidar_angle_max)
    plt.arrow(odometry[0], odometry[1], dx, dy, head_width=0.1, head_length=0.15, fc='grey', ec='grey')
    
    dx = 3 * np.cos(odometry[2]+lidar_angle_min)
    dy = 3 * np.sin(odometry[2]+lidar_angle_min)
    plt.arrow(odometry[0], odometry[1], dx, dy, head_width=0.1, head_length=0.15, fc='grey', ec='grey')
    
    plt.grid(True)
    plt.show()

def lidar_scan_to_3dpc(lidar_ranges):
    '''
    return point cloud in local frame
    '''
    range_mask = (lidar_ranges>lidar_range_min) & (lidar_ranges < lidar_range_max)
    x_offset = 0.13323
    z_offset = 0 #0.51435
    x = lidar_ranges[range_mask] * np.cos(lidar_angles[range_mask]) + x_offset
    y = lidar_ranges[range_mask] * np.sin(lidar_angles[range_mask])
    z = np.full(x.shape, z_offset)
    return np.stack([x,y,z]).T

def odometry_to_transformation(odometry):
    '''
    odometry.shape = n*3
    return wTo shape n*4*4
    '''
    if len(odometry.shape) < 2:
        odometry = np.expand_dims(odometry, 0)

    n = odometry.shape[0]
    cos_theta = np.cos(odometry[:, 2])
    sin_theta = np.sin(odometry[:, 2])

    transformation = np.zeros((n, 4, 4))
    transformation[:, 0, 0] = cos_theta
    transformation[:, 0, 1] = -sin_theta
    transformation[:, 1, 0] = sin_theta
    transformation[:, 1, 1] = cos_theta
    transformation[:, 0, 3] = odometry[:, 0]
    transformation[:, 1, 3] = odometry[:, 1]
    transformation[:, 2, 2] = 1
    transformation[:, 3, 3] = 1
    return transformation

def transformation_to_odometry(Ts):
    '''
    Ts.shape = n x 4 x 4
    '''
    if len(Ts.shape) < 3:
        Ts = np.expand_dims(Ts, 0)
    x = Ts[:, 0, 3]
    y = Ts[:, 1, 3]
    
    # Calculate rotation angle using vectorized operations
    theta_1 = np.arctan2(Ts[:, 1, 0], Ts[:, 0, 0])
    theta_2 = np.arctan2(-Ts[:, 0, 1], Ts[:, 1, 1])
    theta = (theta_1 + theta_2) / 2
    
    return np.stack((x, y, theta), axis=-1)

def diff_transformation(wTsource, wTtarget):
    '''
    input wTsource, wTtarget
    return targetTsource
    '''
    return np.linalg.inv(wTtarget) @ wTsource

def get_T(R,t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    return T

def get_Rt(T):
    T = T.squeeze()
    return T[:3, :3], T[:3, 3]

##### 3. 
def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
      (sx, sy)	start point of ray
      (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))
    

def test_bresenham2D():
    import time
    sx = 0
    sy = 1
    print("Testing bresenham2D...")
    r1 = bresenham2D(sx, sy, 10, 5)
    r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
    r2 = bresenham2D(sx, sy, 9, 6)
    r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
    if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
        print("...Test passed.")
    else:
        print("...Test failed.")

    # Timing for 1000 random rays
    num_rep = 1000
    start_time = time.time()
    for i in range(0,num_rep):
        x,y = bresenham2D(sx, sy, 500, 200)
    print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

##### 4
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                    np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

def test_mapCorrelation():
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    ranges = np.load("test_ranges.npy")

    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    

    
    # xy position in the sensor frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    # convert position in the map frame here 
    Y = np.stack((xs0,ys0))
    
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
        
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)


  
    print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
    ts = tic()
    c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
    toc(ts,"Map Correlation")

    c_ex = np.array([[3,4,8,162,270,132,18,1,0],
        [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
        [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
        [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
        [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
        [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
        [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
        [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
        [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
      
    if np.sum(c==c_ex) == np.size(c_ex):
      print("...Test passed.")
    else:
      print("...Test failed. Close figures to continue tests.")	

    #plot original lidar points
    fig1 = plt.figure()
    plt.plot(xs0,ys0,'.k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading")
    plt.axis('equal')
    
    #plot map
    fig2 = plt.figure()
    plt.imshow(MAP['map'],cmap="hot");
    plt.title("Occupancy grid map")
    
    #plot correlation
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
    ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
    plt.title("Correlation coefficient map")  
    plt.show()
  
if __name__ == '__main__':
    # show_lidar()
    test_mapCorrelation()
    test_bresenham2D()

