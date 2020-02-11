import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from img.interact import sliders
from img.basic import load_image
from img.basic import change_type
from img.basic import change_gamma
from img.basic import display_image
from img.basic import change_contrast
from img.basic import change_vibrance
from img.basic import change_sharpness
from img.basic import change_saturation
from img.basic import change_brightness


def main():

    # load image
    img = load_image('../test.jpg')

    # matplotlib
    fig = plt.figure(figsize=(13,6))

    # set up grids
    gs = gsp.GridSpec(12, 12, figure=fig)

    # display image
    ax = fig.add_subplot(gs[:, 0:7])
    img = load_image('../test.jpg')
    rgb = display_image(img)
    im_obj = ax.imshow(rgb, interpolation='nearest')

    # set up slider ax
    adjustments = ['contrast', 'brightness', 'gamma',
                   'sharpness', 'saturation', 'vibrance']
    slider_count = len(adjustments)
    slider_grid = gsp.GridSpecFromSubplotSpec(
        slider_count, 1, gs[8:, 8:12])
    slider_axes = []
    for (i, text) in enumerate(adjustments):
        slider_ax = fig.add_subplot(slider_grid[i, 0])
        slider_axes.append(slider_ax)

    # change image type for processing
    img = change_type(img, np.float32)

    # set up slider values
    slider_funcs = [change_contrast, change_brightness,
                    change_gamma, change_sharpness,
                    change_saturation, change_vibrance]
    slider_vmins = [0.5, -10, 0.8, -0.5, 0.5, 0.5]
    slider_vmaxs = [2, 10, 1.2, 0.5, 2, 2]
    slider_vinit = [1, 0, 1, 0, 1, 1]

    # connect sliders to image manipulation
    ss = sliders(fig, slider_axes, img, im_obj,
                 slider_funcs, adjustments,
                 slider_vmins, slider_vmaxs,
                 slider_vinit)

    # add a slider for value
    slider_ax = fig.add_subplot(gsp.GridSpecFromSubplotSpec(
        1, 1, gs[5, 8:12])[0, 0], facecolor='lightgoldenrodyellow')
    score_slider = Slider(slider_ax, 'Score', 0, 10, 5)

    # plot the score history
    scores = []
    line_ax = fig.add_subplot(gs[0:3, 8:12])
    score_obj, = line_ax.plot(scores)
    line_ax.grid(linestyle=(0, (10, 10)), color='black', linewidth=0.2)
    line_ax.set_ylim([-1, 11])
    line_ax.set_xlabel('Steps')
    line_ax.set_ylabel('Scores')

    # add next button
    button_ax = fig.add_subplot(gsp.GridSpecFromSubplotSpec(
        1, 1, gs[6, 8:12])[0, 0])
    button = Button(button_ax, 'Bayesian Optimization',
        color='lightgoldenrodyellow', hovercolor='0.975')

    # button performs BO
    def next(event):
        scores.append(score_slider.val)
        score_obj.set_data(np.arange(len(scores)), scores)
        line_ax.set_xlim([-1, len(scores)])
        fig.canvas.draw_idle()
        score_slider.set_val(5)
    button.on_clicked(next)

    plt.show()


if __name__ == '__main__':
    main()
    