import numpy as np
from multiprocessing import Pool
import multiprocessing
import time

IDLE_SLEEP_SEC = 0.1

class GenerativeModel(object):
    def __init__(self, parameter_prior, forward_model, noise_model):
        self.parameter_prior = parameter_prior
        self.forward_model = forward_model
        self.noise_model = noise_model

    def sample(self, batchsize):
        (thetas, weights) = self.parameter_prior.sample(batchsize)
        noiseless = self.forward_model.run_model(thetas,weights)
        images = self.noise_model.add_noise(noiseless)
        return thetas, weights, images

    def sample_eval_batch(self, batchsize, seed):
        np.random.seed(seed)
        (thetas, weights) = self.parameter_prior.sample(batchsize)
        noiseless = self.forward_model.run_model(thetas,weights)
        images = self.noise_model.add_noise(noiseless)
        return thetas, weights, images


class SubprocessSimulator(multiprocessing.Process):
    def __init__(self,gen_function, result_queue, max_size):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.max_size = max_size
        self.gen_function = gen_function

    def run(self):
        proc_name = self.name

        np.random.seed()
        while True:
            if self.result_queue.qsize() < self.max_size:
                self.result_queue.put(self.gen_function())
            else:
                time.sleep(IDLE_SLEEP_SEC)
        return

class MultiprocessGenerativeModel(object):
    def __init__(self, generative_model, n_workers, batchsize):
        self.generative_model = generative_model
        self.queue = multiprocessing.Queue()
        l = lambda: generative_model.sample(batchsize)
        self.workers = [SubprocessSimulator(l, self.queue, 50) for i in range(n_workers)]
        for w in self.workers:
            w.start()

    def sample(self, batchsize):
        return self.queue.get()

    def sample_eval_batch(self, batchsize, seed):
        return self.generative_model.sample_eval_batch(batchsize,seed)

    def stop(self):
        for w in self.workers:
            w.terminate()
