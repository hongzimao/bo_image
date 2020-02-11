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
    for (ax, label, vmin, vmax, vinit) in zip(axs, labels, vmins, vmaxs, vinits):
        ss.append(Slider(ax, label, vmin, vmax, vinit))

    def update(val):
        # load the parameters from outside
        nonlocal img
        nonlocal obj
        nonlocal fig
        nonlocal funcs
        nonlocal ss
        new_img = np.array(img)
        for (s, func) in zip(ss, funcs):
            new_img = func(new_img, s.val)
        obj.set_data(
            display_image(
            change_type(
            clip_image(new_img), np.uint8)))
        fig.canvas.draw_idle()

    for s in ss:
        s.on_changed(update)

    return ss
