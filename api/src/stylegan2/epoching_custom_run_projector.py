# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import dnnlib as dnnlib
import dnnlib.tflib as tflib

import src.stylegan2.projector as projector
import src.stylegan2.pretrained_networks as pretrained_networks
from src.stylegan2.training import dataset
from src.stylegan2.training import misc
from pathlib import Path

# --- Custom Change ---
import pickle

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
            
    print('\r%-30s\r' % '', end='', flush=True)
    
    # --- Custom Change ---
    with open(png_prefix + 'final_latent_code.pkl', 'wb') as out_file:
        pickle.dump(proj.get_dlatents(), out_file)

#----------------------------------------------------------------------------

def project_real_images(Gs, dataset_name, data_dir, num_images, num_snapshots):
    proj = projector.Projector()
    proj.set_network(Gs)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        if not Path('results/image%04d' % image_idx).exists():
            Path('results/image%04d' % image_idx).mkdir()
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('results/image%04d/image%04d-' % (image_idx, image_idx)), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

