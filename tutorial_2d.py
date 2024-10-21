from ransac_2d import Ransac2D
import numpy as np
import matplotlib.pyplot as plt

c0 = 1.0 # slope
c1 = 0.5 # offset

def generate_2d_dataset():
    # Line
    num_points = 100
    xs = np.linspace(0, 1, num_points) # xs

    std = 0.01 # standard deviation for Gaussian noise
    delta = np.random.normal(0, std, len(xs)) # Gaussian noise
    ys = c0 * xs + c1 + delta # ys
    
    # Outliers
    num_outliers = 99
    xs_outliers = np.linspace(0, 1, num_outliers)
    ys_outliers = np.random.uniform(0, 1.5, num_outliers)
    xs = np.concatenate([xs, xs_outliers])
    ys = np.concatenate([ys, ys_outliers])

    return xs, ys

if __name__ == '__main__':
    ## Generate dataset
    xs, ys = generate_2d_dataset()

    ## Visualize dataset
    plt.scatter(xs, ys, c='b')
    
    ## Run RANSAC
    K = 2 # number of sample points (minimum: 2)
    N = 1000 # number of iterations
    t = 0.01 # threshold

    ransac = Ransac2D(K, N, t)

    ransac.run(xs, ys)

    ## Visualize inliers
    inliers = np.array(ransac.best_inliers)
    plt.scatter(inliers[:, 0], inliers[:, 1], color='r')
    plt.plot(xs, ransac.best_c0 * xs + ransac.best_c1, color='g', linestyle='--')
    plt.plot(xs, c0 * xs + c1, color='k', linestyle='-')
    plt.show()


