import nets
import loss_fns
import torch
from torch.autograd import Variable
import torch.optim as optim
import empirical_sim
import prior
import torch.nn as nn
import simple_noise
import generative_model
import time
import sys

# Parameters
batch_size = 256//2#256*10
eval_batch_size = 256
iters_per_eval = 3#10#*4*4
stepsize = 1E-4
_2D = False

if _2D:
    print("Using 2D loss.")

kernel_sigmas = [64.0, 320.0, 640.0, 1920.0]

#
use_cuda = False
warmstart = True

# Construct a generative model
p = prior.UniformCardinalityPrior([0,0,-750], [6400,6400,750], 1000.0, 7000.0, 5)
sim = empirical_sim.EmpiricalSim(64,6400,empirical_sim.load_AS())
noise = simple_noise.EMCCD(100.0)
gen_model = generative_model.GenerativeModel(p,sim,noise)

#Note that queues do not work on OS X.
if sys.platform != 'darwin':
    m = generative_model.MultiprocessGenerativeModel(gen_model,4,batch_size)
else:
    m = gen_model

# Construct the network
# Warmstart?
if warmstart:
    net = torch.load("net_AS")
else:
    net = nets.DeepLoco() if not _2D else nets.DeepLoco(min_coords = [0,0], max_coords = [6400,6400])

if use_cuda:
    net = net.cuda()

theta_mul = Variable(torch.Tensor([1.0,1.0,0.2]))
if use_cuda:
    theta_mul = theta_mul.cuda()

# Takes a CPU batch and converts to CUDA variables
# Also zero/ones the simulated weights
def to_v(d):
    v = Variable(torch.Tensor(d))
    if use_cuda:
        v = v.cuda(async=True)
    return v

def to_variable(theta, weights, images):
    return to_v(theta), to_v(weights).sign_(),to_v(images)

# Loss function
def loss_fn(o_theta, o_w, theta, weights):
    if _2D:
        theta = theta[:,:,:2]
    return loss_fns.multiscale_l1_laplacian_loss(o_theta*theta_mul, o_w,
                                                 theta*theta_mul, weights,
                                                 kernel_sigmas).mean()

# Generate an evaluation batch
(e_theta, e_weights, e_images) = m.sample_eval_batch(eval_batch_size,113)
(e_theta, e_weights, e_images) = to_variable(e_theta, e_weights, e_images)

lr_schedule = [(stepsize, 300), (stepsize/2, 200), (stepsize/4, 100) ,(stepsize/8, 100), (stepsize/16, 100),(stepsize/32, 100),(stepsize/64, 100)]

for stepsize, iters in lr_schedule:
    # Constuct the optimizer
    print("stepsize = ", stepsize)
    optimizer = optim.Adam(net.parameters(),lr=stepsize)
    for i in range(iters):
        iter_start_time = time.time()
        print("iter",i)
        # Compute eval
        (o_theta_e, o_w_e) = net(e_images)
        e_loss = loss_fn(o_theta_e, o_w_e, e_theta,e_weights)
        print("\teval", e_loss.data[0])

        s_time = time.time()
        for batch_idx in range(iters_per_eval):
            print(".")
            (theta, weights, images) = m.sample(batch_size)
            optimizer.zero_grad()
            theta, weights, images = to_variable(theta, weights, images)
            (o_theta, o_w) = net(images)
            train_loss = loss_fn(o_theta,o_w, theta, weights)
            train_loss.backward()
            optimizer.step()
        torch.save(net.cpu(), "net_AS")
        if use_cuda:
            net.cuda()
        print("A:", time.time()-iter_start_time)
