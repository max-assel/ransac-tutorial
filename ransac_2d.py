import numpy as np
import numpy.polynomial.polynomial as poly

class Ransac2D:
    def __init__(self, K, N, t):
        self.K = K
        self.N = N
        self.t = t
        
        self.best_inliers = []
        self.best_c0 = None
        self.best_c1 = None

    ## Sample
    def sample(self, xs, ys):
        ## Randomly select K points
        idx = np.random.choice(len(xs), self.K, replace=False)
        sample_xs = xs[idx]
        sample_ys = ys[idx]

        return sample_xs, sample_ys

    ## Fit
    def least_squares_fit(self, sample_xs, sample_ys):
        # Data matrix
        A = np.vstack([sample_xs, np.ones(len(sample_xs))]).T
        b = sample_ys

        # Least squares solution
        c = np.linalg.lstsq(A, b, rcond=None)[0]

        # print('c: ', c)

        return c[0], c[1]

    ## Fit 
    # def linear_fit(self, sample_xs, sample_ys):

    #     print('sample_xs: ', sample_xs)
    #     print('sample_ys: ', sample_ys)

    #     self.my_linear_fit(sample_xs, sample_ys)

    #     ## Compute linear model parameters
    #     linear = poly.Polynomial.fit(sample_xs, sample_ys, deg=1)

    #     coefs = linear.convert().coef

    #     c0 = coefs[1] # slope
    #     c1 = coefs[0] # offset

    #     # print('poly c0: ' , c0)
    #     # print('poly c1: ' , c1)

    #     return c0, c1
            
    def compute_inliers(self, c0, c1, xs, ys):
        inliers = []
        for x, y in zip(xs, ys):
            y_hat = c0 * x + c1
            if np.abs(y - y_hat) < self.t:
                inliers.append((x, y))
        
        return inliers

    def update_model(self, inliers, c0, c1):
        if len(inliers) > len(self.best_inliers):
            self.best_inliers = inliers
            self.best_c0 = c0
            self.best_c1 = c1

            ## Re-fit model using all inliers
            inliers = np.array(inliers)
            self.best_c0, self.best_c1 = self.least_squares_fit(inliers[:, 0], inliers[:, 1])


    def run(self, xs, ys):
        for i in range(self.N):
            ## Sample
            sample_xs, sample_ys = self.sample(xs, ys)

            ## Fit
            print('Fitting ...')
            c0, c1 = self.least_squares_fit(sample_xs, sample_ys)

            ## Compute inliers
            inliers = self.compute_inliers(c0, c1, xs, ys)

            ## Update best model
            print('Updating ...')
            self.update_model(inliers, c0, c1)
