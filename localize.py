import util
import torch
from torch.autograd import Variable
from PIL import Image
import glob
import numpy as np
import time



visualize = False
_2D = False
use_cuda = False and not visualize

offset = np.array([-46,-47,0]).reshape(1,3) if not _2D else np.array([-(46-4),-(47)]).reshape(1,2)#

net = torch.load("net_AS")
if use_cuda:
    net = net.cuda()

### load contest data
IMAGE_REGEX = './data/test_frames/*.tif'
ACTIVATIONS_CSV = './data/activations.csv'
ACTIVATIONS = np.recfromcsv(ACTIVATIONS_CSV)
ACTIVATIONS.sort(order="frame")
N_TEST_FRAMES = 64#64 if visualize else 19996
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
BATCH_SIZE = 2048
FRAME_ACTIVATIONS = [np.asarray(ACTIVATIONS[ACTIVATIONS['frame']==idx][['xnano','ynano']].tolist()) for idx in range(1,N_TEST_FRAMES+1)]
IMAGE_FILENAMES = glob.glob(IMAGE_REGEX)
IMAGE_FILENAMES.sort()
IMAGE_FILENAMES = IMAGE_FILENAMES[0:N_TEST_FRAMES]

image_chunks = chunks(list(enumerate(IMAGE_FILENAMES)), BATCH_SIZE)

all_points = []
all_weights = []
for chunk in image_chunks:
    print()
    s_time = time.time()
    real_images = torch.stack([torch.Tensor(np.array(Image.open(filename)).astype(np.float32)) for (idx,filename) in chunk])
    real_images = (real_images-700)/(90/15) #162)/1.5#/(700))/(89/15) #???
    batchsize = real_images.size(0)
    if use_cuda:
        real_images = real_images.cuda()

    print("load time: ", time.time() - s_time)
    s_time = time.time()
    (o_theta, o_w) = net(Variable(real_images, volatile=True))
    print("net_time: ", time.time()-s_time)
    s_time = time.time()

    o_theta, o_w = o_theta.data, o_w.data
    theta_mul = torch.Tensor([1.0,1.0,0.2]).cuda() if use_cuda else torch.Tensor([1.0,1.0,0.2])
    (points,weights) = util.fast_batch_cwa(o_theta, o_w, 0.001, 100, 0.3, theta_mul = theta_mul )

    print("post_time: ", time.time()-s_time)
    all_points += points
    all_weights += weights


if not visualize:
    points = all_points
    to_write = [(f_idx+1,(np.array(ps)+offset)) for (f_idx, ps) in enumerate(points)]

    rows = []
    idx = 1
    for (f_idx, ps) in to_write:
        for p in ps:
            rows.append([idx,f_idx]+p.tolist())
            idx += 1
    if _2D:
        np.savetxt("2d_loc.csv", rows, fmt = "%d, %d, %10.5f, %10.5f")
    else:
        np.savetxt("3d_loc.csv", rows, fmt = "%d, %d, %10.5f, %10.5f, %10.5f")
    torch.save(points, "points")
else:
    import pylab
    pylab.ion()
    for img_idx in range(64):
        pylab.figure()
        pylab.imshow(real_images[img_idx], extent=(0,6400,6400,0))
        pylab.scatter(points[img_idx][:,0], points[img_idx][:,1], marker='x',color="red",s = (weights[img_idx])*100.0) #,s = o_w[b_idx,:].data.cpu()*100.0)
        if len( FRAME_ACTIVATIONS[img_idx])> 0:
            pylab.scatter(FRAME_ACTIVATIONS[img_idx][:,0], FRAME_ACTIVATIONS[img_idx][:,1], marker='o',color="white",facecolor="none") #,s = o_w[b_idx,:].data.cpu()*100.0)
