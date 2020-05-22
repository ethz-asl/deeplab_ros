# ROS Wrapper for DeepLab

This is a package for using DeepLab models with ROS. DeepLab is a state-of-the-art deep learning architecture for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image. For more information about DeepLab, please visit [this link](https://github.com/tensorflow/models/tree/master/research/deeplab).

Code for running inference is based on the following [Colab notebook](https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb).

<p align="center">
<img src="https://raw.githubusercontent.com/ethz-asl/deeplab_ros/master/doc/deeplab_ros.gif" width="600">
</p>

## Citing

If you use the code for your research, please cite this work as:
```bibtex
@misc{grinvald2018deeplabros,  
  author={Margarita Grinvald},
  title={ROS Wrapper for DeepLab},
  year={2018}
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ethz-asl/deeplab_ros}},
}
```

## Getting started
Clone this repository to the `src` folder of your catkin workspace, build your workspace and source it.

```bash
cd <catkin_ws>/src
git clone git@github.com:ethz-asl/deeplab_ros.git
catkin build
source <catkin_ws>/devel/setup.bash
```

## Example usage
An example launch file is included processing a sequence from the [Freiburg RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).

```bash
cd <catkin_ws>/src/deeplab_ros
chmod +x scripts/download_freiburg_rgbd_example_bag.sh 
scripts/download_freiburg_rgbd_example_bag.sh
roslaunch deeplab_ros freiburg.launch
```


## Usage

#### Parameters:

* **`~rgb_input`** [_string_]

    Topic name of the input RGB stream.

    Default: `"/camera/rgb/image_color"`

* **`~model`** [_string_]

    Name of the backbone network used for inference. List of available models: {"mobilenetv2_coco_voctrainaug", "mobilenetv2_coco_voctrainval", "xception_coco_voctrainaug", "xception_coco_voctrainval"}.
    If the specified model file doesn't exist, the node automatically downloads the file.

    Default: `"mobilenetv2_coco_voctrainaug"`
    
* **`~visualize`** [_bool_]

    If true, the segmentation result overlaid on top of the input RGB image is published to the `~segmentation_viz` topic.

    Default: `true`
    
        
#### Topics subscribed:

* topic name specified by parameter **`~rgb_input`** (default: **`/camera/rgb/image_color`**) [_sensor_mgs/Image_]

    Input RGB image to be processed.

    
#### Topics published:

* **`~segmentation`** [_sensor_mgs/Image_]

    Segmentation result.


* **`~segmentation_viz`** [_sensor_mgs/Image_]

    Visualization-friendly segmentation result color coded with the PASCAL VOC 2012 color map overlaid on top of the input RGB image.


