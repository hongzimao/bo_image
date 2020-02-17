# Enhancing Images with Bayesian Optimization

At the age of Instagram, no photo can go public before extensive post processing. From common users to professional photographers, everyone is trying to differentiate themsleves by finding a unique processing vibe. To explore different possibilities, one must step out from her familiar processing pipeline (e.g., always sharpen and intensify the iamge).

This simple tool allows you to explore and enhance images by only ranking how you think of the processing result. The program will systematically try out different settings based on your feedback. Hopefully, it will find a good combination of settings through a few iterations. During each optimization step, you are still free to adjust any setting through the interactive interface.

## Prerequisite

python 3.7.3, opencv-python 4.2.0, numpy 1.16.3, scipy 1.3.0, matplotlib 3.0.3, sklearn 0.22

## Example

In `/src`, run
```
python3 main.py --input_image ../data/example.jpg
```

At every step, provide feedback for the adjustment by dragging the `Score` slider. You can also change any of the processing sliders if you like. Click on the `Optimize` button to log the results and perform one step of Bayesian Optimization. Click `Save` to save the current processing result. The screenshot below shows the optimization process.

![Screenshot](https://github.com/hongzimao/bo_image/blob/master/data/example_screenshot.png)
