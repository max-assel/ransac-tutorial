import numpy as np
import numpy.polynomial.polynomial as poly

class Ransac3D:
    def __init__(self, K, N, t):
        self.K = K
        self.N = N
        self.t = t
        
        self.best_inliers = []
        self.best_c0 = None
        self.best_c1 = None
        self.best_c2 = None        

    ## Sample
    def sample(self, xs, ys, zs):
        ## Randomly select K points
        idx = np.random.choice(len(xs), self.K, replace=False)
        sample_xs = xs[idx]
        sample_ys = ys[idx]
        sample_zs = zs[idx]

        return sample_xs, sample_ys, sample_zs

    ## Fit
    def least_squares_fit(self, sample_xs, sample_ys, sample_zs):
        # Data matrix
        A = np.vstack([sample_xs, sample_ys, np.ones(len(sample_xs))]).T
        b = sample_zs

        # Least squares solution
        c = np.linalg.lstsq(A, b, rcond=None)[0]

        # print('c: ', c)

        return c[0], c[1], c[2]

    # ## Fit 
    # def planar_fit(self, sample_xs, sample_ys):

    #     print('sample_xs: ', sample_xs)
    #     print('sample_ys: ', sample_ys)

    #     pass
            
    def compute_inliers(self, c0, c1, c2, xs, ys, zs):
        inliers = []
        for x, y, z in zip(xs, ys, zs):
            z_hat = c0 * x + c1 * y + c2 
            if np.abs(z - z_hat) < self.t:
                inliers.append((x, y, z))
        
        return inliers

    def update_model(self, inliers, c0, c1, c2):
        if len(inliers) > len(self.best_inliers):
            self.best_inliers = inliers
            self.best_c0 = c0
            self.best_c1 = c1
            self.best_c2 = c2

            ## Re-fit model using all inliers
            inliers = np.array(inliers)
            self.best_c0, self.best_c1, self.best_c2 = self.least_squares_fit(inliers[:, 0], inliers[:, 1], inliers[:, 2])


    def run(self, xs, ys, zs):
        for i in range(self.N):
            ## Sample
            sample_xs, sample_ys, sample_zs = self.sample(xs, ys, zs)

            ## Fit
            print('Fitting ...')
            c0, c1, c2 = self.least_squares_fit(sample_xs, sample_ys, sample_zs)

            ## Compute inliers
            inliers = self.compute_inliers(c0, c1, c2, xs, ys, zs)

            ## Update best model
            print('Updating ...')
            self.update_model(inliers, c0, c1, c2)
