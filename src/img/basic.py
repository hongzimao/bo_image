import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    return img


def change_type(img, dtype):
    return img.astype(dtype)


def display_image(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    rgb_img = np.stack([r, g, b], axis=2)
    plt.imshow(rgb_img, interpolation='nearest')
    plt.show()


def change_contrast(img, alpha):
    assert(img.dtype != np.uint8)
    img = np.clip(img * alpha, 0, 255)
    return img


def change_brightness(img, beta):
    assert(img.dtype != np.uint8)
    img = np.clip(img + beta, 0, 255)
    return img


def change_gamma(img, gamma):
    assert(img.dtype != np.uint8)
    img = np.clip((img / 255) ** gamma * 255, 0, 255)
    return img


def change_sharpness(img, amount):
    assert(img.dtype != np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.addWeighted(img, 1 + amount, blur, -amount, gamma=0)
    img = np.clip(img, 0, 255)
    return img


def change_saturation(img, amount):
    assert(img.dtype != np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= amount
    img = np.clip(img, 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = np.clip(img, 0, 255)
    return img


def change_vibrance(img, amount):
    assert(img.dtype != np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    amount = np.ones(hsv.shape[0:2]) * amount
    amount **= (1 - hsv[:, :, 1])
    hsv[:, :, 1] *= amount
    img = np.clip(img, 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = np.clip(img, 0, 255)
    return img


def write_image(img, path):
    cv2.imwrite(path, img)
