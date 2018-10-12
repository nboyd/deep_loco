import numpy as np
import time

# A noise model adds noise to a batch of images.
# Noise models should return images such that the noise has
# roughly zero mean.

# TODO: Add random (per-image) parameter versions of these noise models

# From SMLM2016 website...
class EMCCD(object):
    def __init__(self, noise_background = 0.0, quantum_efficiency=0.9, read_noise=74.4, spurious_charge=0.0002,em_gain=300.0, baseline=100.0, e_per_adu=45.0):
        self.qe = quantum_efficiency
        self.read_noise = read_noise
        self.c = spurious_charge
        self.em_gain = em_gain
        self.baseline = baseline
        self.e_per_adu = e_per_adu
        self.noise_bg = noise_background

    def add_noise(self, photon_counts):
        s_time = time.time()
        #print()
        n_ie = np.random.poisson(self.qe*(photon_counts+self.noise_bg) + self.c)
        #print("poisson: ", time.time()-s_time)
        n_oe = np.random.gamma(n_ie+0.001, scale=self.em_gain)
        #print("gamma: ", time.time()-s_time)
        n_oe = n_oe + np.random.normal(0.0,self.read_noise,n_oe.shape)
        #print("normal: ", time.time()-s_time)
        ADU_out = (n_oe/self.e_per_adu).astype(int) + self.baseline
        #print("cast: ", time.time()-s_time)
        #print()
        return self.center(np.minimum(ADU_out,65535))

    def gain(n):
        return n.qe*n.em_gain/n.e_per_adu

    def mean(n):
        return (n.noise_bg*n.qe + n.c)*n.em_gain/n.e_per_adu + n.baseline

    def center(self, img):
        return (img - self.mean())/self.gain()

class GaussianNoise(object):
    def __init__(self, std = 0.3):
        self.std = std

    def add_noise(self, photon_counts):
        return ((photon_counts + np.random.normal(0.0, self.std, photon_counts.shape)))/self.std

class RandomNoise(object):
    def __init__(self, noise_models):
        self.noise_models = noise_models

    def add_noise(self, photon_counts):
        model = np.random.choice(self.noise_models)
        return model.add_noise(photon_counts)
