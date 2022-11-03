from src.service.db import GeneratorZsDb
from src.generator.generator import Generator
from pathlib import Path
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import re

home_path = str(Path.home())

database = GeneratorZsDb(home_path + '/api/src/persistance')


class GeneratorService:
    def __init__(self):
        self.home_path = str(Path.home())
        self.generator = Generator(network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

    @staticmethod
    def get_ids():
        ids = []
        zs = []
        for id, z in database.fetch_all():
            ids.append(id)
            zs.append(z)
        return ids, zs

    def get_images_from_database(self, from_id, to_id):
        ids, zs = GeneratorService.get_ids()
        actual_ids = []
        imgs_bytes = []
        for id, z in zip(ids, zs):
            if id >= from_id and id <= to_id:
                actual_ids.append(id)
                image = self.generate_face(id)
                imgs_bytes.append(self.generator.Image_to_bytes(image))
        return imgs_bytes, actual_ids

    def generate_face(self, id: int):
        print("Generating image...")
        z = database.fetch_z_by_id(id=id)
        image = self.generator.generate_image_from_z(z)
        return Image.fromarray(image)

    def generate_random_images(self, qty: int):
        print("Generating random images...")
        imgs_bytes = []
        zs = []
        for i in range(qty):
            seed = np.random.randint(30000)
            image, z = self.generator.generate_random_image(seed)
            imgs_bytes.append(self.generator.Image_to_bytes(image))
            zs.append(z)
        return imgs_bytes, zs

    def generate_transition(self, id_img1: int, id_img2: int, qty: int = None):
        print("Generating transition...")
        z1 = database.fetch_z_by_id(id=id_img1)
        z2 = database.fetch_z_by_id(id=id_img2)

        if qty is None:
            imgs, zs = self.generator.generate_transition(z1, z2)
        else:
            imgs, zs = self.generator.generate_transition(z1, z2, qty)
        imgs_bytes = [self.generator.Image_to_bytes(img) for img in imgs]

        return imgs_bytes, zs

    def base64_to_latent(self, base64str):
        print("Locating face in latent space...")
        base64_data = re.sub('^data:image/.+;base64,', '', base64str)
        img_bytes = base64.b64decode(base64_data)
        data = BytesIO(img_bytes)
        imgs, zs = self.generator.img_to_latent(Image.open(data))
        imgs_bytes = [self.generator.Image_to_bytes(img) for img in imgs]
        return imgs_bytes, zs

    def change_features(self, id_img: int, features: dict):
        print("Modifying face...")
        z = database.fetch_z_by_id(id=id_img)
        new_img, new_z = self.generator.change_features(z, features)
        img_bytes = self.generator.Image_to_bytes(new_img)
        return img_bytes, new_z[0]

    def save_image(self, z):
        print("Saving image...")
        database.insert_z(z)
        new_id = len(database.fetch_all())
        return new_id

    def mix_styles(self, id1, id2):
        z1 = database.fetch_z_by_id(id=id1)
        z2 = database.fetch_z_by_id(id=id2)
        imgs, zs = self.generator.mix_styles(z1, z2)
        imgs_bytes = [self.generator.Image_to_bytes(img) for img in imgs]
        return imgs_bytes, zs

