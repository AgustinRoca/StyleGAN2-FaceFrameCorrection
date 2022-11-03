import sys
from os import mkdir, path
from pathlib import Path
import io
from src.face_frame import face_frame_correction

home_path = str(Path.home())
sys.path.insert(0, home_path + "/api/src/stylegan2")

import dnnlib
from src.stylegan2 import pretrained_networks
from dnnlib import tflib
from src.stylegan2 import dataset_tool
from src.stylegan2 import epoching_custom_run_projector

import numpy as np
from PIL import Image
import base64
from tqdm import tqdm
import pickle
from src.generator.align_face import align_face


class Generator:

    def __init__(self, network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl'):
        self.fps = 20
        self.result_size = 640
        self.network_pkl = network_pkl
        self.latent_vectors = self.get_control_latent_vectors('src/generator/stylegan2directions')

        self.Gs, self.noise_vars, self.Gs_kwargs = self.load_model()

    def load_model(self):
        _G, _D, Gs = pretrained_networks.load_networks(self.network_pkl)
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        return Gs, noise_vars, Gs_kwargs

    def get_control_latent_vectors(self, path):
        files = [x for x in Path(path).iterdir() if str(x).endswith('.npy')]
        latent_vectors = {f.name[:-4]:np.load(f) for f in files}
        return latent_vectors

    def generate_random_image(self, rand_seed):
        '''returns the image and its latent code'''
        src_latents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in [rand_seed])
        z = self.Gs.components.mapping.run(src_latents, None)
        # z_avg = self.Gs.get_var('dlatent_avg')

        z_avg = [9.28829536e-02,  4.44930047e-02,  1.75801829e-01, -7.59617761e-02,
 -1.42774582e-01,  2.15757936e-02, -7.72733241e-02, -8.58754367e-02,
  1.46154195e-01, -1.13609456e-01,  2.29243040e-01, -3.13183665e-03,
 -5.95544279e-03,  1.18608497e-01, -1.29310340e-01,  1.00871488e-01,
 -3.20591033e-03, -2.43059173e-02,  9.38473940e-02,  4.07060832e-02,
 -6.36393726e-02, -1.01074576e-03, -1.16058908e-01,  5.69779202e-02,
  2.01586992e-01,  4.73712504e-01, -2.98747085e-02, -3.53142321e-02,
  1.89392656e-01, -3.19052935e-02,  9.94438231e-02,  4.91961837e-03,
  2.41641745e-01, -5.21609187e-02,  1.09912813e-01,  1.87420473e-01,
  8.29494521e-02,  4.65616584e-02, -1.09433658e-01,  1.81137681e-01,
  1.41442314e-01,  3.49077076e-01, -8.03427398e-03,  1.56212747e-01,
  3.54236007e-01, -3.19860913e-02,  1.60225093e-01,  1.96288526e-02,
  2.58380234e-01, -8.20623487e-02,  1.44453064e-01,  1.01666451e-01,
  3.42814505e-01, -5.90227097e-02, -2.52769887e-02,  4.37498242e-02,
  8.75642300e-02,  1.77401438e-01, -1.79897964e-01, -1.10987172e-01,
  1.23952270e-01,  9.46108550e-02, -8.80334079e-02,  5.87446131e-02,
  3.52484822e-01,  1.55435830e-01,  1.18199855e-01,  1.98041931e-01,
 -4.16216925e-02,  4.79252785e-02, -1.17687196e-01,  3.12857211e-01,
 -1.44437075e-01,  1.46079808e-02,  3.07293832e-02,  2.15120345e-01,
  6.81429505e-02,  2.24278867e-01,  4.40626964e-03,  2.94559360e-01,
  1.66019797e-03,  5.67742214e-02,  8.41903985e-02,  1.59137011e-01,
  8.21952373e-02,  2.28891343e-01,  1.24274962e-01,  1.86963305e-01,
 -1.16458826e-01,  2.63504565e-01,  4.67748404e-01, -1.30166873e-01,
 -4.07232232e-02,  4.02291194e-02,  3.16444993e-01,  1.22022629e-03,
 -8.78280215e-03, -9.91897285e-03,  5.15452623e-01,  1.22251898e-01,
  1.13079898e-01,  9.68366861e-03,  1.05182743e-02,  9.80355591e-02,
  2.17795402e-01,  1.71064734e-01,  3.64652365e-01,  1.99257851e-01,
 -3.25690731e-02,  1.98756799e-01,  1.01452708e-01,  3.32868606e-01,
  1.20538458e-01,  4.08407062e-01, -8.64214599e-02,  2.63415396e-01,
  7.01134801e-02,  1.28153741e-01,  1.58935130e-01, -1.18322000e-02,
  2.73714304e-01, -8.93704221e-02,  4.99394834e-02,  1.03364646e-01,
 -1.04774617e-01, -2.40193382e-02,  9.66360271e-02,  1.95008725e-01,
  1.71969295e-01, -2.00226530e-02,  2.88615882e-01,  1.42723143e-01,
 -8.95538926e-02,  1.65939808e-01, -5.94716966e-02,  2.54610568e-01,
  2.47283280e-02,  1.06786296e-01,  4.45023239e-01,  2.29900926e-02,
  1.20995507e-01, -3.86784673e-02,  2.76407748e-02,  7.56703988e-02,
  9.54320282e-03,  1.71767414e-01, -6.12324476e-03,  8.20922554e-01,
  2.89800595e-02, -9.86350924e-02,  3.37426484e-01, -8.84753317e-02,
 -5.12907654e-02,  2.13945836e-01, -1.31220102e-01, -3.17265093e-03,
 -9.42373201e-02,  2.82817692e-01, -6.17964566e-02, -3.09260190e-03,
  1.47985071e-01,  6.01789206e-02,  2.61123329e-02, -8.47694576e-02,
  2.21754089e-02,  8.90364647e-02,  1.54573768e-01,  1.54623419e-01,
  2.73957532e-02,  6.42251149e-02,  9.20453370e-02,  5.09756446e-01,
  5.02602756e-02, -7.02050030e-02, -3.92222032e-02,  1.68808684e-01,
  9.55543667e-02,  5.40113688e-01,  8.01889300e-02, -7.04348236e-02,
  2.30105668e-01,  9.95282531e-02,  6.13948554e-02,  1.24455757e-01,
 -8.05358961e-03,  4.54206765e-02,  2.37128615e-01,  2.20747650e-01,
  4.09807414e-02, -1.12708591e-01,  1.78295001e-01,  2.34072775e-01,
  6.94813728e-02,  1.19812250e-01,  3.12556982e-01, -6.63028285e-02,
 -1.97966397e-02,  4.52881753e-01, -3.75787914e-02,  1.59160525e-01,
  1.86184034e-01, -5.52046746e-02,  3.11710089e-02,  3.06921631e-01,
  1.83373690e-02,  2.31311321e-01, -3.60896736e-02,  1.44345224e-01,
  8.28133166e-01,  2.83546478e-01,  9.95307267e-02, -1.16284579e-01,
  2.04264835e-01, -2.27554828e-01,  1.02306306e-01, -2.40995586e-02,
  9.69578922e-02,  2.54059225e-01, -5.72264865e-02, -7.73982555e-02,
  4.03178096e-01, -3.10703553e-03,  1.33526564e-01,  4.13239896e-01,
  3.50062907e-01,  2.50291824e-02,  2.70393074e-01,  3.31723809e-01,
 -2.90195644e-03, -8.15991312e-03, -5.19659370e-02,  1.95924640e-01,
  3.43684673e-01,  1.79953367e-01,  8.07352543e-01,  9.04203579e-03,
  2.64214352e-02,  2.59683341e-01,  2.98537314e-02,  6.75268099e-02,
 -2.33601034e-03,  1.99408293e-01, -1.27340719e-01,  9.85689834e-02,
  4.76223379e-02, -4.61745188e-02,  3.03163752e-02,  2.79872417e-02,
  2.45040268e-01,  3.26501131e-01, -3.57468724e-02,  1.14937350e-01,
  9.74076688e-02,  2.63080627e-01,  2.92999864e-01,  2.41966486e-01,
  1.88552499e-01, -7.68197700e-02,  2.79222786e-01,  5.89840934e-02,
  1.52695492e-01, -1.94433331e-02,  4.61954176e-01,  2.46546552e-01,
 -6.16166666e-02,  1.17392182e-01,  5.99621981e-02,  1.39062613e-01,
  1.25477552e-01,  2.24084198e-01,  1.59644455e-01,  1.74129605e-02,
  2.79583037e-02,  4.81164157e-02,  1.78805754e-01,  2.51432806e-02,
  4.22968119e-02,  1.65830687e-01,  4.35781777e-02, -8.50072205e-02,
  3.38507742e-02,  2.44587272e-01,  5.83560243e-02,  7.41212294e-02,
  1.36812672e-01, -2.35738792e-02,  1.18148208e-01,  7.08480179e-02,
  4.37389731e-01, -2.42968053e-02, -1.00226596e-01,  3.48088741e-02,
 -5.85323200e-02, -5.09456843e-02,  6.89145178e-02,  6.45897910e-02,
  1.83425277e-01,  2.22199440e-01, -6.46788925e-02, -1.09110281e-01,
  1.25397205e-01,  5.58336377e-02,  2.13560462e-03,  9.03552026e-02,
 -1.05490118e-01,  1.94729492e-01, -3.52492481e-02,  1.18815482e-01,
  1.16203181e-01, -3.71904410e-02,  2.66023368e-01, -2.89239064e-02,
  8.71634185e-02, -2.12596916e-02,  1.23818278e-01,  9.67195332e-02,
  5.46977103e-01, -1.59883201e-02, -8.67757872e-02,  1.31143779e-01,
  2.90087044e-01, -9.07666683e-02,  6.75601214e-02,  1.71099499e-01,
  7.64759853e-02,  2.68181920e-01,  2.49293745e-02,  4.87653725e-02,
  4.67907488e-01, -1.43141896e-02,  1.01432621e-01,  2.19855726e-01,
 -8.78572762e-02,  5.27605295e-01,  1.89506084e-01,  2.34247833e-01,
  3.58624101e-01,  2.11986780e-01,  1.23461701e-01,  1.56731918e-01,
  2.42886379e-01,  1.94570020e-01,  3.73836756e-02,  2.02176213e-01,
  2.76287109e-01,  3.67532790e-01, -1.03121497e-01,  1.58818245e-01,
 -8.53630006e-02,  9.20942426e-02,  2.21435651e-02,  1.06310636e-01,
  1.63818859e-02, -1.97469965e-02, -3.62060294e-02,  2.08687633e-01,
 -5.82576916e-02, -7.66599849e-02, -2.76911259e-02,  1.52412429e-01,
  3.64866443e-02,  8.41373578e-02,  1.07947379e-01,  9.81532931e-02,
  1.30017236e-01,  2.52841741e-01,  1.21452339e-01,  1.27608120e-01,
 -5.89363500e-02,  2.41968423e-01,  3.83654162e-02, -7.13027492e-02,
  1.47014230e-01, -4.39807773e-02,  1.89126339e-02,  1.53668702e-01,
  1.30419463e-01, -6.17328286e-03,  1.25298843e-01,  9.29070264e-02,
  3.56976032e-01, -5.89192957e-02, -4.65279222e-02,  3.31002772e-02,
  5.35374582e-02,  1.86607242e-01,  2.71232754e-01,  1.63697124e-01,
  2.64235556e-01,  1.19628191e-01, -9.70020592e-02, -1.60498202e-01,
  2.58300938e-02, -8.80375803e-02, -7.43998140e-02,  9.36091393e-02,
  2.14031562e-02,  6.03337288e-01,  9.90867019e-02,  1.09761775e-01,
  7.55933896e-02,  3.27285826e-01,  1.12956867e-01,  6.74808174e-02,
  2.10269183e-01, -1.04341790e-01,  8.57647881e-02, -1.72370225e-02,
  2.34821409e-01, -1.27850063e-02,  1.09542370e-01, -7.01197237e-02,
  5.19675203e-02,  2.11024284e-03, -6.24671876e-02,  5.17478138e-02,
  1.27206787e-01,  8.38514268e-02,  3.29422504e-02, -6.31067157e-03,
  1.37038469e-01, -1.51713356e-01, -1.04964487e-01, -7.22974539e-03,
 -4.25912738e-02, -4.82889526e-02, -1.98576450e-02, -9.15165395e-02,
  2.94061512e-01,  1.74358487e-02,  1.65009230e-01,  3.99844706e-01,
  7.17592835e-02, -1.21585540e-01, -3.75219136e-02, -5.89125790e-03,
  1.19914144e-01,  2.94425786e-01,  1.24903992e-01,  4.00079936e-02,
  3.66320312e-02,  2.18684040e-02,  4.46286649e-02,  2.59265780e-01,
  2.05250293e-01,  1.20070741e-01, -1.02307737e-01,  1.59908950e-01,
  4.04856920e-01,  1.44359946e-01,  3.76234800e-02, -1.25184476e-01,
  1.64749652e-01,  1.95523500e-01, -1.58461809e-01,  2.14728445e-01,
 -1.18398033e-01,  3.96567643e-01,  1.30406052e-01, -1.18046999e-04,
 -1.05923116e-02, -6.85571134e-02, -8.33642483e-02,  1.50763571e-01,
  8.05545598e-04,  3.65073621e-01,  3.54211032e-03,  1.26016140e-01,
  2.38059267e-01,  3.19308549e-01, -6.45162761e-02,  2.41103500e-01,
  1.06915295e-01,  1.06855489e-01,  1.99437678e-01, -1.35614574e-01,
  1.39946952e-01,  1.25257596e-01, -8.19748864e-02,  1.21687353e-01,
  1.28524214e-01, -3.25499475e-02,  4.93912101e-01, -4.10717651e-02,
  1.21706657e-01,  1.78169012e-01, -1.35973319e-02, -2.16441341e-02,
 -4.80329990e-02,  1.47165060e-02, -1.32124037e-01,  1.13103233e-01,
  6.55654371e-02,  2.97989309e-01,  4.16606665e-03,  5.60403109e-01,
  6.02781773e-03,  2.05371574e-01,  2.83563197e-01,  2.45739117e-01,
  1.03726961e-01,  1.09772220e-01,  4.26042080e-02,  4.85264882e-02,
  1.04114711e-01,  1.06961176e-01,  2.44579054e-02,  1.31025016e-02,
  1.25602260e-01, -5.96663207e-02, -4.14476432e-02,  2.01182723e-01]

        for i in range(9):
            z[0][i] = (z[0][i]-z_avg)*0.7 + z_avg
        random_image = self.generate_image_from_z(z)
        image = Image.fromarray(random_image)
        return image, z

    def generate_image_from_z(self, z):
        images = self.Gs.components.synthesis.run(z, **self.Gs_kwargs)
        return images[0]

    def linear_interpolate(self, code1, code2, alpha):
        return code1 * alpha + code2 * (1 - alpha)

    def img_to_latent(self, img: Image):
        aligned_imgs_path = Path('aligned_imgs')

        if not aligned_imgs_path.exists():
            aligned_imgs_path.mkdir()
        img_name = 'image0000'
        result = align_face(img)
        if result is None:
            return None, None
        result.save(aligned_imgs_path/('aligned_'+img_name+'.png'))
        dataset_tool.create_from_images('datasets_stylegan2/custom_imgs', aligned_imgs_path, 1)
        epoching_custom_run_projector.project_real_images(self.Gs, 'custom_imgs', 'datasets_stylegan2', 1, 2)
        
        all_result_folders = list(Path('results').iterdir())
        all_result_folders.sort()
        last_result_folder = all_result_folders[-1]
        all_step_pngs = [x for x in last_result_folder.iterdir() if x.name.endswith('png') and 'image{0:04d}'.format(0) in x.name]
        all_step_pngs.sort()

        target_image = Image.open(all_step_pngs[-1]).resize((self.result_size, self.result_size))
        best_aproximation = Image.open(all_step_pngs[-2]).resize((self.result_size, self.result_size))
        latent_code = self.get_final_latents()

        zs, images = face_frame_correction(target_image, latent_code, self.Gs, self.Gs_kwargs)
        
        return images, zs

    def get_final_latents(self):
        all_results = list(Path('results/').iterdir())
        all_results.sort()
        
        last_result = all_results[-1]

        latent_files = [x for x in last_result.iterdir() if 'final_latent_code' in x.name]
        latent_files.sort()
        
        all_final_latents = []
        
        for file in latent_files:
            with open(file, mode='rb') as latent_pickle:
                all_final_latents.append(pickle.load(latent_pickle))
    
        return all_final_latents[0]

    def generate_transition(self, z1, z2, num_interps=50):
        step_size = 1.0/num_interps
    
        all_imgs = []
        all_zs = []
        
        amounts = np.arange(0, 1, step_size)
        
        for alpha in tqdm(amounts):
            interpolated_latent_code = self.linear_interpolate(z2, z1, alpha)
            image = self.generate_image_from_z(interpolated_latent_code)
            interp_latent_image = Image.fromarray(image).resize((self.result_size, self.result_size))
            all_imgs.append(interp_latent_image)
            all_zs.append(interpolated_latent_code)
        return all_imgs, all_zs
        
    def change_features(self, z, features_amounts_dict: dict):
        modified_latent_code = np.array(z)
        for feature_name, amount in features_amounts_dict.items():
            modified_latent_code += self.latent_vectors[feature_name] * amount
        image = self.generate_image_from_z(modified_latent_code)
        latent_img = Image.fromarray(image).resize((self.result_size, self.result_size))
        return latent_img, modified_latent_code

    def mix_styles(self, z1, z2):
        z1_copy = z1.copy()
        z2_copy = z2.copy()
        z1[0][6:] = z2_copy[0][6:]
        z2[0][6:] = z1_copy[0][6:]
        image1 = Image.fromarray(self.generate_image_from_z(z1)).resize((self.result_size, self.result_size))
        image2 = Image.fromarray(self.generate_image_from_z(z2)).resize((self.result_size, self.result_size))
        return [image1, image2], [z1, z2]



    def Image_to_bytes(self, img):
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        return encoded_img
