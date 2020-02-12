import numpy as np


def update_image(img, funcs, vals):
    # manipulate image from scratch
    # TODO: make every operator invertible
    new_img = np.array(img)

    # sequentially apply operators
    for (f, v) in zip(funcs, vals):
        new_img = f(new_img, v)

    return new_img
