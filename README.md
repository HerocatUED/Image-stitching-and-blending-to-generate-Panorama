# Image-stitching-and-blending-to-generate-Panorama

## Quick Start
Step 1. 

Put your images in $./inputs/panoramas/NameOfPanorama$ folder, with which you want to generate panorama. 

Step 2. 

Run ***OpenCV_panoramas.py*** under the root folder. The command should be  \
```python OpenCV_panoramas.py```

You can find the output in $./output/panoramas$ folder.

This should work in most situations. But if this does not work right, you can try to run ***panoramas_modes.py***. You can choose which technique to use in ***panoramas_modes.py***. If you don't want to choose the techniques, it will use all technique and you can classify them by the suffix of file name.
## Further Learning
This programm uses many techniques to gennerate panorama, including feature detctors&descriptors such as Harris Corner, SIFT, SURF, AKAZE, ORB, BRISK, and blending skills such as alpha blending, multiband blending, poisson blending, pyramid blending.

If you want to learn more about the technique, you can find details of implement in ***report.pdf***. 
