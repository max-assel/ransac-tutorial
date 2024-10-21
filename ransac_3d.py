import numpy as np
import numpy.polynomial.polynomial as poly

class Ransac3D:
    def __init__(self, K, N, t):
        self.K = K
        self.N = N
        self.t = t
        
        self.best_inliers = []
        self.best_m = None
        self.best_b = None

    ## Sample
    def sample(self, xs, ys):
        ## Randomly select K points
        idx = np.random.choice(len(xs), self.K, replace=False)
        sample_xs = xs[idx]
        sample_ys = ys[idx]

        return sample_xs, sample_ys


    def fit_plane(points):
        """Fit a plane to a set of 3D points.

        Args:
            points (numpy.ndarray): An array of shape (n, 3) representing n 3D points.

        Returns:
            numpy.ndarray: The coefficients of the plane equation (a, b, c, d),
                        where the equation is ax + by + cz + d = 0.
        """
        pass

    ## Fit 
    def planar_fit(self, sample_xs, sample_ys):

        print('sample_xs: ', sample_xs)
        print('sample_ys: ', sample_ys)

        pass
            
    def compute_inliers(self, m, b, xs, ys):
        inliers = []
        for x, y in zip(xs, ys):
            y_hat = m * x + b
            if np.abs(y - y_hat) < self.t:
                inliers.append((x, y))
        
        return inliers

    def update_model(self, inliers, m, b):
        if len(inliers) > len(self.best_inliers):
            self.best_inliers = inliers
            self.best_m = m
            self.best_b = b

            ## Re-fit model using all inliers
            inliers = np.array(inliers)
            self.best_m, self.best_b = self.linear_fit(inliers[:, 0], inliers[:, 1])


    def run(self, xs, ys):
        for i in range(self.N):
            ## Sample
            sample_xs, sample_ys = self.sample(xs, ys)

            ## Fit
            print('Fitting ...')
            m, b = self.linear_fit(sample_xs, sample_ys)

            ## Compute inliers
            inliers = self.compute_inliers(m, b, xs, ys)

            ## Update best model
            print('Updating ...')
            self.update_model(inliers, m, b)
