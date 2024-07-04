# License-Plate-Recognition-YOLOv7-and-CNN

This project is for the ultimate manner of identifying the License Plate. Combining YOLOv7 object detection, Hough transform alignment, and CNN character recognition

## Table of contents
* [1. How to use](#1-How-to-use)
* [2. Result](#2-result)

## 1. How to use

* Remember to set up neccesary libraries in `requirements.txt` 
* Download the model used for YOLOv7 model `LP_detect_yolov7_500img.pt` and CNN model `weight.h5` in Git RELEASES and put them in the right path like in the code
* To test on image/video, run `main_image.py`/ `main_video.py`. Remember to change the path of image/video. I don't provide videos for testing, but you can record it yourself. **1920x1080 pixels, 24 fps recommend**
* In `data` folder you can find `data.yaml` needed for YOLOv7 training and folder `test` including test images. Feel free to use it
* `doc` images for documents
* `src` folder are codes for CNN model. put the CNN model here
* `utils` and `models` are for YOLOv7. They're a part of original YOLO. However, don't care about them, you can use YOLOv7 to derectly detect License Plates with `detect.py`. I have changed the code a lot compared to the original one. It's now much easier to use
* `Preprocess.py`, `utils_LP.py` and `vid2img.py` are util files. Spend time to explore them.
* `yolo-v7-license-plate-detection.ipynb` is the training of YOLOv7

## 2. Result

<p align="center"><img src="doc/final_result 2.png" width="500">            <img src="doc/LP_detected_img.png" width="500"></p>
<p align="center"><i>Figure. Final results </i></p>
