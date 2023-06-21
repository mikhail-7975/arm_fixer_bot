import cv2
import segmentation_models_pytorch as smp
import numpy as np


class Preprocessor:
    def __init__(self) -> None:
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    def preprocessing(self, img_path):
        image = cv2.imread(img_path)
        img_preprocessed = self.preprocessing_fn(image)
        img_preprocessed = cv2.resize(img_preprocessed, (224,224))
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        img_preprocessed = np.transpose(img_preprocessed, (0, 3, 1, 2))

        return image, img_preprocessed.astype(np.float32)