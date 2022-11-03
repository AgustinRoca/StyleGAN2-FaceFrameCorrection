from glob import glob
from src.generator.align_face import align_face
import numpy as np
import cv2
from PIL import Image
import torch
from src.nets.MobileNetV2_unet import MobileNetV2_unet
from torchvision import transforms
from tqdm import tqdm


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    state_dict = torch.load('./src/checkpoints/model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def segmentate_image(img, transform, model):
    torch_img = transform(img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))

    # Forward Pass
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    return mask

def face_frame_variation(img1, img2, transform, model):
    segmentation_1 = segmentate_image(img1, transform, model)
    segmentation_2 = segmentate_image(img2, transform, model)
    difference_array = np.subtract(segmentation_1, segmentation_2)
    difference_array = np.abs(difference_array)
    result = len(np.where(difference_array == 2)[0])
    indexes_to_check = np.where(difference_array == 1)
    for i, j, k in zip(indexes_to_check[0], indexes_to_check[1], indexes_to_check[2]):
        classification_1 = segmentation_1[i][j][k]
        classification_2 = segmentation_2[i][j][k]
        if classification_1 == 0 or classification_2 == 0:
            result += 1
        elif (classification_1 == 1 and classification_2 == 2) or (classification_1 == 2 and classification_2 == 1):
            result += 0.2
        else:
            print(classification_1, classification_2)
    result = (result / (224*224)) * 100
    return result, segmentation_1, segmentation_2

def face_frame_correction(target_image, latent_code, Gs, Gs_kwargs):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    steps = 750
    initial_noise_factor = 0.005
    noise_ramp_length = 0.75
    dlatent_avg_samples = 10000
    latent_samples = np.random.RandomState(123).randn(dlatent_avg_samples, *Gs.input_shapes[0][1:])
    dlatent_samples = Gs.components.mapping.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
    dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
    dlatent_std = (np.sum((dlatent_samples - dlatent_avg) ** 2) / dlatent_avg_samples) ** 0.5

    image = Gs.components.synthesis.run(latent_code, **Gs_kwargs)
    image = Image.fromarray(image[0])
    min_frame, _, _ = face_frame_variation(target_image, image, transform, model)
    print('Starting frame', min_frame)
    style = (latent_code.copy())[0][6:]
    latent_codes = []
    images = []
    latent_codes.append(0)
    latent_codes.append(latent_code.copy())
    images.append(target_image.copy())
    images.append(image)

    for i in tqdm(range(steps)):
        old_code = latent_code.copy()
        t = i/10000
        noise_strength = dlatent_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        noise = np.random.normal(size=latent_code.shape) * noise_strength
        latent_code += noise
        latent_code[0][6:] = style
        image = Gs.components.synthesis.run(latent_code, **Gs_kwargs)
        image = Image.fromarray(image[0])
        frame_variation, _, _ = face_frame_variation(target_image, image, transform, model)
        if frame_variation < min_frame:
            print('Correction made', frame_variation)
            min_frame = frame_variation
            latent_codes.append(latent_code.copy())
            images.append(image)
        else:
            latent_code = old_code

    return latent_codes, images
