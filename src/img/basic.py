import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    return img


def change_type(img, dtype):
    return img.astype(dtype)


def display_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb
    # import matplotlib.pyplot as plt
    # plt.imshow(rgb, interpolation='nearest')
    # plt.show()


def clip_image(img):
    img = np.clip(img, 0, 255)
    return img


def change_contrast(img, alpha):
    assert(img.dtype != np.uint8)
    return img * alpha


def change_brightness(img, beta):
    assert(img.dtype != np.uint8)
    return img + beta


def change_gamma(img, gamma):
    assert(img.dtype != np.uint8)
    img = np.clip(img, 0, np.inf)
    return (img / 255) ** gamma * 255


def change_sharpness(img, amount):
    assert(img.dtype != np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.addWeighted(img, 1 + amount, blur, -amount, gamma=0)
    return img


def change_saturation(img, amount):
    assert(img.dtype != np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= amount
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def change_vibrance(img, amount):
    assert(img.dtype != np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    amount = np.ones(hsv.shape[0:2]) * amount
    amount **= (1 - hsv[:, :, 1])
    hsv[:, :, 1] *= amount
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def write_image(img, path):
    cv2.imwrite(path, img)
