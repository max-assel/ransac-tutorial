from ransac_3d import Ransac3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

m = 1.0 # slope
b = 0.5 # bias

def generate_3d_dataset():
    num_points = 10

    # Plane
    xs = np.linspace(0, 1, num_points)
    ys = np.linspace(0, 1, num_points)
    xgrid, ygrid = np.meshgrid(xs, ys)

    std = 0.1    # standard deviation for Gaussian noise
    delta = np.random.normal(0, std, np.shape(xgrid))
    zgrid = m * xgrid + b * ygrid + delta

    xs = xgrid.flatten()
    ys = ygrid.flatten()
    zs = zgrid.flatten()

    # print('np.shape(xgrid): ', np.shape(xgrid))
    # print('xgrid: ', xgrid)

    # Outliers
    num_outliers = 10
    xs_outliers = np.linspace(0, 1, num_outliers)
    ys_outliers = np.linspace(0, 1, num_outliers)
    xgrid_outliers, ygrid_outliers = np.meshgrid(xs_outliers, ys_outliers)

    zgrid_outliers = np.random.uniform(0, 1, np.shape(xgrid_outliers))

    xs_outliers = xgrid_outliers.flatten()
    ys_outliers = ygrid_outliers.flatten()
    zs_outliers = zgrid_outliers.flatten()

    xs = np.concatenate([xs, xs_outliers])
    ys = np.concatenate([ys, ys_outliers])
    zs = np.concatenate([zs, zs_outliers])

    return xs, ys, zs

if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ## Generate dataset
    # xs, ys = generate_2d_dataset()
    xs, ys, zs = generate_3d_dataset()

    ## Visualize dataset
    # plt.scatter(xs, ys, c='b')
    ax.scatter(xs, ys, zs, c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    ## Run RANSAC
    K = 3 # number of sample points (minimum: 2)
    N = 1000 # number of iterations
    t = 0.01 # threshold

    ransac = Ransac3D(K, N, t)

    ransac.run(xs, ys)

    ## Visualize inliers
    inliers = np.array(ransac.best_inliers)
    plt.scatter(inliers[:, 0], inliers[:, 1], color='r')
    plt.plot(xs, ransac.best_m * xs + ransac.best_b, color='g', linestyle='--')
    plt.plot(xs, m * xs, color='k', linestyle='-')
    plt.show()


