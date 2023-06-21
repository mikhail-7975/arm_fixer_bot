import numpy as np
import cv2
import onnxruntime as ort

from preprocessing import Preprocessor


class Model:
    def __init__(self) -> None:
        self.DEVICE = 'cpu'
        self.preproc = Preprocessor()
        self.ort_session = ort.InferenceSession("arm_fractures_unet_resnext50.onnx", providers=['CPUExecutionProvider'])

    def inference(self, img_path):
        image, input_image = self.preproc.preprocessing(img_path)
        h, w = image.shape[:2]

        outputs = self.ort_session.run(
            None,
            {'x.1': input_image}
            )
        out = outputs[0].squeeze().round()
        
        out = cv2.resize(out, (w, h)).astype('uint8')
        _, thresh = cv2.threshold(out, 0.5, 255, cv2.THRESH_BINARY_INV)
        pure = np.zeros(thresh.shape).astype('uint8')
        out3 = cv2.merge((thresh, pure, pure))

        res = cv2.addWeighted(image, 0.5, out3, 0.5, 0)
        
        return res
