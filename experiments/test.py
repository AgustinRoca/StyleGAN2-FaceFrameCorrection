from glob import glob
from align_face import align_face
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_face_in_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image)
    # pil_img = align_face(pil_img)
    return pil_img

def segmentate_image(image_path, transform, model):
    pil_img = get_face_in_image(image_path)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(args.device)

    # Forward Pass
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    return mask

def face_frame_variation(original_path, generated_path, transform, model):
    original_result = segmentate_image(original_path, transform, model)
    generated_result = segmentate_image(generated_path, transform, model)
    difference_array = np.subtract(original_result, generated_result)
    difference_array = np.abs(difference_array)
    result = len(np.where(difference_array == 2)[0])
    indexes_to_check = np.where(difference_array == 1)
    for i, j, k in zip(indexes_to_check[0], indexes_to_check[1], indexes_to_check[2]):
        original_classification = original_result[i][j][k]
        generated_classification = generated_result[i][j][k]
        if original_classification == 0 or generated_classification == 0:
            result += 1
        elif (original_classification == 1 and generated_classification == 2) or (original_classification == 2 and generated_classification == 1):
            result += 0.2
        else:
            print(original_classification, generated_classification)
    result = (result / (224*224)) * 100
    return result, original_result, generated_result


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    parser.add_argument('--original-data-folder', type=str, default='./data/original',
                        help='name of the data folder (default: ./data/original)')
    parser.add_argument('--generated-data-folder', type=str, default='./data/generated',
                        help='name of the data folder (default: ./data/generated)')
    parser.add_argument('--corrected-data-folder', type=str, default='./data/corrected',
                        help='name of the data folder (default: ./data/corrected)')
    parser.add_argument('--pre-trained', type=str, default='./checkpoints/model.pt',
                        help='path of pre-trained weights (default: ./checkpoints/model.pt)')

    args = parser.parse_args()
    args.device = torch.device("cpu")

    original_image_files = sorted(glob('{}/*.png'.format(args.original_data_folder)))
    generated_image_files = sorted(glob('{}/*.png'.format(args.generated_data_folder)))
    corrected_image_files = sorted(glob('{}/*.png'.format(args.corrected_data_folder)))
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    print(len(original_image_files), 'files in folder', args.original_data_folder)
    print(len(generated_image_files), 'files in folder', args.generated_data_folder)
    print(len(corrected_image_files), 'files in folder', args.corrected_data_folder)

    fig = plt.figure()
    for a, (original_path,generated_path, corrected_path) in enumerate(zip(original_image_files, generated_image_files, corrected_image_files)):
        result, original_result, generated_result = face_frame_variation(original_path,generated_path, transform, model)
        result_str_1 = f'{result:.2f}%'
        result, original_result, corrected_result = face_frame_variation(original_path,corrected_path, transform, model)
        result_str_2 = f'{result:.2f}%'
        print(result_str_1, result_str_2)

        ax = plt.subplot(3, 2, 1)
        ax.axis('off')
        ax.imshow(get_face_in_image(original_path))
        ax = plt.subplot(3, 2, 2)
        ax.axis('off')
        ax.imshow(original_result.squeeze())
        # ax.text(200, 265, result_str_1, fontsize=8)
        ax = plt.subplot(3, 2, 3)
        ax.axis('off')
        ax.imshow(get_face_in_image(generated_path))
        ax = plt.subplot(3, 2, 4)
        ax.axis('off')
        ax.imshow(generated_result.squeeze())
        ax.text(80, 250, result_str_1, fontsize=8)
        ax = plt.subplot(3, 2, 5)
        ax.axis('off')
        ax.imshow(get_face_in_image(corrected_path))
        ax = plt.subplot(3, 2, 6)
        ax.axis('off')
        ax.imshow(corrected_result.squeeze())
        ax.text(80, 250, result_str_2, fontsize=8)
    # fig.subplots_adjust(top=0.1, bottom=0)
    plt.show()

