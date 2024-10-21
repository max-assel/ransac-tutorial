from ransac_3d import Ransac3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

c0 = 1.0 # ???
c1 = 0.5 # ???
c2 = 0.2 # ???

def generate_3d_dataset():
    num_points = 10

    # Plane
    xs = np.linspace(0, 1, num_points)
    ys = np.linspace(0, 1, num_points)
    xgrid, ygrid = np.meshgrid(xs, ys)

    std = 0.1    # standard deviation for Gaussian noise
    delta = np.random.normal(0, std, np.shape(xgrid))
    zgrid = c0 * xgrid + c1 * ygrid + c2 + delta

    xs = xgrid.flatten()
    ys = ygrid.flatten()
    zs = zgrid.flatten()

    # print('np.shape(xgrid): ', np.shape(xgrid))
    # print('xgrid: ', xgrid)

    # Outliers
    num_outliers = 0
    xs_outliers = np.linspace(0, 1, num_outliers)
    ys_outliers = np.linspace(0, 1, num_outliers)
    xgrid_outliers, ygrid_outliers = np.meshgrid(xs_outliers, ys_outliers)

    zgrid_outliers = np.random.uniform(0, 2, np.shape(xgrid_outliers))

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
    ax.scatter3D(xs, ys, zs, color='b', alpha=0.50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    
    ## Run RANSAC
    K = 3 # number of sample points (minimum: 2)
    N = 1000 # number of iterations
    t = 0.01 # threshold

    ransac = Ransac3D(K, N, t)

    ransac.run(xs, ys, zs)

    ## Visualize inliers
    inliers = np.array(ransac.best_inliers)
    ax.scatter3D(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='r', alpha=1.0)
    
    # Create a grid of x and y values
    x_plane = np.linspace(0, 1, 10)
    y_plane = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x_plane, y_plane)
    best_Z = ransac.best_c0 * X + ransac.best_c1 * Y + ransac.best_c2
    ax.plot_surface(X, Y, best_Z, alpha=0.5, color='g')

    Z = c0 * X + c1 * Y + c2
    ax.plot_surface(X, Y, Z, alpha=0.5, color='k')

    plt.show()


