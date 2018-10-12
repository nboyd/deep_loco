import scipy.interpolate
from PIL import Image
import numpy as np
import glob
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STACK_NPZ_FILENAME_DH = os.path.join(SRC_DIR, 'dh_zstack.npz')
DATA_DIR_DH = "data/sequence-as-stack-Beads-DH-Exp-as-list/"

STACK_NPZ_FILENAME_AS = os.path.join(SRC_DIR, 'as_zstack.npz')
DATA_DIR_AS = "data/z-stack-Beads-AS-Exp-as-list/"

STACK_NPZ_FILENAME_2D = os.path.join(SRC_DIR, '2d_zstack.npz')
DATA_DIR_2D = "data/z-stack-Beads-2D-Exp-as-list/"

def load_2D():
    return dict(np.load(STACK_NPZ_FILENAME_2D))

def load_AS():
    # the NPZ object can't be pickled causing all manner of hell
    # with pywren, so we convert it to a dict
    return dict(np.load(STACK_NPZ_FILENAME_AS).items())

def load_DH():
    return dict(np.load(STACK_NPZ_FILENAME_DH).items())

def load_name(n):
    if n == 'emp_as':
        return load_AS()
    elif n == 'emp_dh':
        return load_DH()
    elif n == 'emp_2d':
        return load_2D()
    else:
        raise ValueError()


### We determined that our simulator actually
### has a subtle error. To match with SMLM contest 3d data,
### this is the scale factor we should apply to RECOVERED x, y, and z

SIM_ERR_SCALE = np.array([64.0/65, 64.0/65, 1.0])


#TODO: HARDCODES 151 FRAMES FROM Z = -750 to 750
def preprocess_images(data_dir):
    """
    Load activations and images

    does not subtract off mean
    returns z location independently
    """

    filenames = glob.glob(data_dir+"/*.tif")
    filenames.sort()
    imgs = np.array([np.array(Image.open(img)).astype(np.float64) for img in filenames])

    activations = np.loadtxt(data_dir+"/activations.csv", delimiter=",")
    z_vals = np.linspace(-750,750,151)
    z_stack = [activations[activations[:,4]==z]  for z in [x for x in z_vals]]
    return {'imgs' : imgs,
            'z_vals': z_vals,
            'z_stack' : np.array(z_stack)}

#TODO: Hardcodes many parameters
class EmpiricalSim(object):
    ## Correct value to match SMLM 2016 data of multiplier is multiplier=1./(30000 * 6. )
    def __init__(self, n_pixels_frame, frame_width_nm, stack_data, multiplier=1.0/250000):#STACK_NPZ_FILENAME):
        """
        Use saved data from a z-stack to generate simulated STORM images
        """
        noise_mean = 100.0 # ? % ? % ? %
        self.multiplier = multiplier#1/100000.0
        self.psf_width_nm = 400
        self.n_pixels_frame = n_pixels_frame
        self.frame_width_nm = frame_width_nm
        self.z_depth = 1500.0 # this is hardcoded below...

        #stack_data = np.load(STACK_NPZ_FILENAME)
        imgs = stack_data['imgs'] - noise_mean
        z_stack = stack_data['z_stack']
        z_vals = stack_data['z_vals']

        splines = []
        offsets = []
        psfs = []
        w_ests = []
        for f_idx in range(151):
            xs,ys = z_stack[f_idx][:,2],z_stack[f_idx][:,3]
            ### linear approx
            #### ZERO OUT BOUNDARY....
            img = imgs[f_idx]
            img[0,:] = 0.0
            img[-1,:] = 0.0
            img[:,0] = 0.0
            img[0,-1] = 0.0
            spline = scipy.interpolate.RectBivariateSpline([r*100 for r in range(150)],[r*100 for r in range(150)],img, kx=1, ky=1)
            ### The contest z-stack images are improperly normalized.
            splines.append(spline)
            xs = np.delete(xs,[1,5])
            ys = np.delete(ys,[1,5])
            offsets.append((xs,ys))
        self.splines = splines
        self.offsets = offsets

    def draw(self, x,y,z, image, w):
        """
        Render a point source at (x, y, z) nm with weight w
        onto the passed-in image.
        """

        #x,y,z in NM
        #### coordinate transform...

        assert(z <= 750 and z >= -750)
        new_pixel_locations = np.linspace(0.0, self.frame_width_nm, self.n_pixels_frame)
        z_scaled = (z + 750)/1500*150
        z_low = int(np.floor(z_scaled))
        z_high = int(np.ceil(z_scaled))
        max_dist = 2700.0
        x_filter = abs(new_pixel_locations - x) < max_dist
        y_filter = abs(new_pixel_locations - y) < max_dist
        x_l,x_u = x_filter.nonzero()[0].min(), x_filter.nonzero()[0].max()+1
        y_l,y_u = y_filter.nonzero()[0].min(), y_filter.nonzero()[0].max()+1
        for p_idx in [np.random.randint(4)]:
            spline_l = self.splines[z_low]
            spline_h = self.splines[z_high]
            alpha = 1.0 - (z_scaled - z_low)
            # print(alpha)
            beta = 1.0 - alpha
            alpha *= self.multiplier*w
            beta *= self.multiplier*w
            (xs,ys) = self.offsets[z_low]
            x_s, y_s = xs[p_idx], ys[p_idx]
            ### filtered locations?
            f_y = (new_pixel_locations +(y_s-y))[y_filter]
            f_x = (new_pixel_locations +(x_s-x))[x_filter]
            # we should be integrating here...? oh well.
            d_image = spline_l(f_y,f_x)
            image[y_l:y_u, x_l:x_u] += alpha*d_image

            (xs,ys) = self.offsets[z_high]
            x_s, y_s = xs[p_idx], ys[p_idx]

            f_y = (new_pixel_locations +(y_s-y))[y_filter]
            f_x = (new_pixel_locations +(x_s-x))[x_filter]
            # we should be integrating here...? oh well.
            d_image = spline_h(f_y,f_x)
            image[y_l:y_u, x_l:x_u] += beta*d_image
            #clip z to 10 nm?
        return image

#TODO: Should we multiprocess this here?
#TODO: WE ARE HARDCODING 1 PIXEL = 100 NM.
    def run_model(self, thetas, weights):
        """
        Generate a batch empirically.
        """

        # These are numpy arrays
        batch_size = thetas.shape[0]
        MAX_N = thetas.shape[1]
        assert thetas.shape == (batch_size, MAX_N, 3)
        assert weights.shape == (batch_size, MAX_N)

        images = np.zeros((batch_size, self.n_pixels_frame,self.n_pixels_frame))

        for b_idx in range(batch_size):
            image = np.zeros((self.n_pixels_frame,self.n_pixels_frame))
            for (w, (x,y,z)) in zip(weights[b_idx], thetas[b_idx]):
                if w != 0.0:
                    self.draw(x, y, z, image, w)
            images[b_idx] = image
        return images

def create_z(STACK_NPZ_FILENAME,DATA_DIR ):
    res = preprocess_images(DATA_DIR)
    np.savez_compressed(STACK_NPZ_FILENAME, **res)

if __name__ == "__main__":
    import pylab
    sim = EmpiricalSim(64,6400,load_AS())
    #img =
    for (idx,z) in enumerate(np.linspace(-500,500,9)):
        fig = pylab.figure(frameon=False)
        fig.set_size_inches(1,1)
        ax = pylab.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        pylab.imshow((sim.draw(3200,3200,z, np.zeros((64,64)), 1500.0))[:,:], interpolation="none", cmap="inferno")
        pylab.savefig("z_stack_{}.png".format(idx),dpi=512)

    for (idx,x) in enumerate(np.linspace(2700.0,3700.0,3)):
        fig = pylab.figure(frameon=False)
        fig.set_size_inches(1,1)
        ax = pylab.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        pylab.imshow((sim.draw(x,3200,0.0, np.zeros((64,64)), 1500.0))[:,:], interpolation="none", cmap="inferno")
        pylab.savefig("x_{}.png".format(idx),dpi=512)

    dense = np.random.randn(64,64)
    for i in range(1000):
        (x,y,z) = np.random.uniform(0, 6400), np.random.uniform(0, 6400), np.random.uniform(-750, 750)
        sim.draw(x,y,z,dense, np.random.uniform(500, 1500))

    fig = pylab.figure(frameon=False)
    fig.set_size_inches(1,1)
    ax = pylab.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pylab.imshow(dense, interpolation="none", cmap="inferno")
    pylab.savefig("ultradense.png".format(idx),dpi=512)
    #pylab.show()
