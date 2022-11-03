from glob import glob
from align_face import align_face
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
import os


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to('cpu')
    state_dict = torch.load('./checkpoints/model.pt', map_location='cpu')
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
    torch_img = torch_img.to('cpu')

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
    import matplotlib.pyplot as plt

    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    base_path = './data/directions'
    paths = {'age': 3, 'gender': 3, 'horizontal': 3, 'vertical': 3, 'mouthopen': 10, 'eyesopen': 10, 'smile': 2}
    subfolders = sorted([f.path for f in os.scandir(base_path) if f.is_dir()])

    fig = plt.figure()
    ax = plt.subplot(111)
    for path in subfolders:
        print(path)
        faces = sorted([f.path for f in os.scandir(path) if f.is_dir()])
        yss = []
        xs = np.linspace(-1*paths[path.split('/')[-1]], paths[path.split('/')[-1]], 20)
        xs = xs.tolist()
        xs.append(0)
        xs = sorted(xs)
        for face in faces:
            target_path = f'{face}/target.png'
            target = get_face_in_image(target_path)
            images = sorted([f.path for f in os.scandir(face) if f.is_file()])
            images.remove(target_path)
            ys = []
            for image in images:
                result, original_result, generated_result = face_frame_variation(target_path, image, transform, model)
                print(path.split('/')[-1], image, result)
                ys.append(result)
                if len(ys) == 10:
                    ys.append(0)
            yss.append(ys)
        
        mean_ys_axis = np.mean(yss, axis=0)
        plt.plot(xs, mean_ys_axis, label=path.split('/')[-1])
        plt.scatter(xs, mean_ys_axis)
    plt.xlabel('Direction')
    plt.ylabel('Variation')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
 

