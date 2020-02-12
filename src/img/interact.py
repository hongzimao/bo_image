import numpy as np
import matplotlib.pyplot as plt
from img.basic import clip_image
from img.basic import change_type
from img.basic import display_image
from matplotlib.widgets import Slider, Button


def slider(fig, ax, img, func, obj, label, vmin, vmax, vinit):
    '''
    fig: plt.figure
    ax: plt.axes
    img: cv2.imread
    func: image manipulation functions in src.img.basic
    obj: plt.imshow object
    '''

    s = Slider(ax, label, vmin, vmax, vinit)
    last_val = vinit

    def update(val):
        # load the parameters from outside
        nonlocal img
        nonlocal obj
        nonlocal fig
        nonlocal last_val
        # modify the image value itself
        img[:] = func(img, val - last_val)
        last_val = val
        obj.set_data(
            display_image(
            change_type(
            clip_image(img), np.uint8)))
        fig.canvas.draw_idle()

    s.on_changed(update)

    return s


def sliders(fig, axs, img, obj, funcs, labels, vmins, vmaxs, vinits):
    ss = []
    for (ax, label, vmin, vmax, vinit) in zip(
        axs, labels, vmins, vmaxs, vinits):
        # add a slider
        ss.append(Slider(ax, label, -1, 1, 0))

    def update(val):
        # load the parameters from outside
        nonlocal img
        nonlocal obj
        nonlocal fig
        nonlocal funcs
        nonlocal ss
        # manipulate image from scratch
        # TODO: make every operator invertible
        new_img = np.array(img)
        for (s, func, vmin, vmax, vinit) in zip(
            ss, funcs, vmins, vmaxs, vinits):

            # map to actual value
            # TODO: map the multiplication relation differently
            if s.val >= 0:
                v = vinit + s.val * (vmax - vinit)
            else:
                v = vinit - s.val * (vmin - vinit)

            # modify figure
            new_img = func(new_img, v)
        # update image
        obj.set_data(
            display_image(
            change_type(
            clip_image(new_img), np.uint8)))
        # matplotlib update
        fig.canvas.draw_idle()

    for s in ss:
        s.on_changed(update)

    return ss
