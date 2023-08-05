import os
import cv2
import numpy as np
import albumentations as albu
from model import preprocessing_fn
CLASSES = ['bird','ground animal','curb','fence','guard rail','barrier','wall','bike lane','crosswalk - plain',
'curb cut','parking','pedestrian area','rail track','road','service lane','sidewalk','bridge','building',
'tunnel','person','bicyclist','motorcyclist','other rider','lane marking - crosswalk','lane marking - general',
'mountain','sand','sky','snow','terrain','vegetation','water','banner','bench','bike rack','billboard',
'catch basin','cctv camera','fire hydrant','junction box','mailbox','manhole','phone booth','pothole',
'street light','pole','traffic sign frame','utility pole','traffic light','traffic sign (back)','traffic sign (front)',
'trash can','bicycle','boat','bus','car','caravan','motorcycle','on rails','other vehicle','trailer',
'truck','wheeled slow','car mount','ego vehicle','unlabeled']

class Dataset:
    """TrayDataset. Read images, apply augmentation and preprocessing transformations."""

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids_x = sorted(os.listdir(images_dir))
        self.ids_y = sorted(os.listdir(masks_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Get the file path of the image
        img_file_path = self.ids_x[i]

        # Open the image file and transform it into a numpy array
        img = cv2.imread(img_file_path)

        # Perform any necessary transformations
        # If your get_preprocessing() function returns a transform function that
        # takes a numpy array and returns a numpy array, you can apply it here
        img = get_preprocessing(preprocessing_fn)(img)

        # Do the same thing for the mask (ground truth data)
        mask_file_path = self.ids_y[i]
        mask = cv2.imread(mask_file_path, 0)  # Assuming mask is grayscale
        mask = get_preprocessing(preprocessing_fn)(mask)

        return img, mask

    def __len__(self):
        return len(self.ids_x)


def get_training_augmentation():
    train_transform = [
        albu.Resize(256, 320, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)],
            p=0.9),
        albu.IAAAdditiveGaussianNoise(p=0.2),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(256, 320)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
