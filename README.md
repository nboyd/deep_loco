# deep_loco

A sample implementation of [deeploco](https://www.biorxiv.org/content/early/2018/02/16/267096).

train_script.py trains a neural net to do localization using simulated data generated
 from a z-stack from the 2016 SMLM challenge.

You'll need a machine with a (reasonably) powerful GPU to train quickly (set use_cuda=True).

To try this on a new dataset you'll need a z-stack (see empirical_sim.py) and to
make sure that the simulated data looks as similar as possible to the real data.
This could be quite difficult: you'll need to adjust many hardcoded values in empirical_sim.py,
as well as the generative model settings (in train_script.py). You might also need to modify
the learning rate schedule of the network.

localize.py gives an example of how to use a pretrained network to do localization.
