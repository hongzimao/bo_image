# Enhancing Images with Bayesian Optimization

At the age of Instagram, no photo can go public before extensive post processing. From common users to professional photographers, everyone is trying to differentiate themsleves by finding an unique processing vibe. To explore different possibilities, one must step out from her familiar processing pipeline (e.g., always sharpen and intensify the iamge).

This simple tool allows you to explore and enhance the image by only ranking how you think the result performs. The program will systematically try out different settings based on your feedback. Hopefully, it will find a good combination of settings through a few iterations. During each optimization step, you are still free to adjust any setting through the interactive interface.

## Example

In `/src`, run
```
python3 main.py --input_image ../data/example.jpg
```

At every step, provide feedback for the adjustment by dragging the `Score` slider. You can change any of the processing slider if you like. Click on the `Optimize` button to log the results and perform one step of Bayesian Optimization. Click `Save` to save the current image processing result. The screenshot below shows the optimization process.

![Screenshot](https://github.com/hongzimao/bo_image/blob/master/data/example_screenshot.png)
