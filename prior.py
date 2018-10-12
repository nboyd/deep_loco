import numpy as np # Should we be using pytorch here?

# Describes a method for generating parameters (fluorophore spatial locations) and weights (fluorophore intensities)
class UniformCardinalityPrior(object):
    def __init__(self, MIN_SPATIAL, MAX_SPATIAL, MIN_W, MAX_W, N):
        self.MIN = np.array(MIN_SPATIAL)
        self.MAX = np.array(MAX_SPATIAL)
        self.MIN_W = MIN_W
        self.MAX_W = MAX_W
        self.N = N

    def sample(self, batch_size):
        weights = np.random.uniform(self.MIN_W, self.MAX_W, (batch_size,self.N))
        #### each frame gets a number of sources that is uniform in {0, ..., N}
        n_sources = np.random.randint(0,self.N+1, batch_size)
        for b_idx in range(batch_size):
            weights[b_idx,:n_sources[b_idx]] = 0.0
        thetas = np.random.uniform(low = self.MIN, high = self.MAX, size = (batch_size,self.N,len(self.MIN)))
        return thetas, weights
