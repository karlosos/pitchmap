from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx

import numpy as np
import tensorflow as tf
import cv2
import segmentation_models as sm
from pitchmap.homography.keypoints_utils import _points_from_mask


class KeypointDetectorModel:
    """Class for Keras Models to predict the keypoint in an image. These keypoints can then be used to
    compute the homography.

    Arguments:
        backbone: String, the backbone we want to use
        model_choice: The model architecture. ('FPN','Unet','Linknet')
        num_classes: Integer, number of mask to compute (= number of keypoints)
        input_shape: Tuple, shape of the model's input
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(
        self,
        backbone="efficientnetb3",
        model_choice="FPN",
        num_classes=29,
        input_shape=(320, 320),
    ):

        self.input_shape = input_shape
        self.classes = [str(i) for i in range(num_classes)] + ["background"]
        self.backbone = backbone

        n_classes = len(self.classes)
        activation = "softmax"

        if model_choice == "FPN":
            self.model = sm.FPN(
                self.backbone,
                classes=n_classes,
                activation=activation,
                input_shape=(input_shape[0], input_shape[1], 3),
                encoder_weights="imagenet",
            )
        else:
            self.model = None
            print("{} is not used yet".format(model_choice))

        self.preprocessing = _build_keypoint_preprocessing(input_shape, backbone)

    def __call__(self, input_img):

        img = self.preprocessing(input_img)
        pr_mask = self.model.predict(np.array([img]))
        return pr_mask

    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "from Imagenet"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )


def _build_keypoint_preprocessing(input_shape, backbone):
    """Builds the preprocessing function for the Field Keypoint Detector Model.

    """
    sm_preprocessing = sm.get_preprocessing(backbone)

    def preprocessing(input_img, **kwargs):

        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing


def main():
    import matplotlib.pyplot as plt
    import imutils

    kp_model = KeypointDetectorModel(
        backbone="efficientnetb3",
        num_classes=29,
        input_shape=(320, 320),
    )

    WEIGHTS_NAME = "./models/FPN_efficientnetb3_0.0001_8_427.h5"
    kp_model.load_weights(WEIGHTS_NAME)

    image_path = "./data/homography_test_images/BaltykStarogard (7).jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pr_mask = kp_model(image)
    src_points, dst_points = _points_from_mask(pr_mask[0])

    # Plot image with detected points
    plt.imshow(image)
    for point in src_points:
        plt.plot(point[0], point[1], marker='o', color='red')
    plt.show()

    # Plot model with detected points
    model_path = "./data/pitch_model.jpg"
    model = cv2.imread(model_path)
    model = imutils.resize(model, width=600)

    plt.imshow(model)
    for point in dst_points:
        plt.plot(point[0], point[1], marker='o', color='red')
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()