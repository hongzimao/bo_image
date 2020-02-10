import numpy as np
from src.img.basic import load_image
from src.img.basic import change_type
from src.img.basic import change_gamma
from src.img.basic import display_image
from src.img.basic import change_contrast
from src.img.basic import change_vibrance
from src.img.basic import change_sharpness
from src.img.basic import change_saturation
from src.img.basic import change_brightness



def main():
    img = load_image('./test.jpg')
    display_image(img)
    img = change_type(img, np.float32)
    # img = change_brightness(img, 10)
    # img = change_contrast(img, 1.2)
    # img = change_gamma(img, 0.8)
    # img = change_sharpness(img, 1)
    # img = change_saturation(img, 2)
    # img = change_vibrance(img, 1.5)
    img = change_type(img, np.uint8)
    display_image(img)


if __name__ == '__main__':
    main()
    