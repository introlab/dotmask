# DOTMask

**D**ynamic **O**bject **T**racking and **Mask**ing is a simple and modular visual SLAM pipeline that to improves both localization and mapping in dynamic environments for mobile robotics. 

## Requirements

Python 3.6.9 (& 2.7), Cuda 10.2, PyTorch, ROS Melodic, RTAB-Map

*For Mask R-CNN install TensorFlow 1.3 and Keras 2.0.8*

*For better performances,install RTAB-Map with recomended/optional dependencies*


## Installation

1. Project dependencies

    ```bash
    sudo apt-get install python3-pip python3-yaml python3-catkin-pkg-modules python3-rospkg-modules python3-empy
    pip3 install torch torchvision cython
    pip3 install opencv-python pillow pycocotools matplotlib scikit-learn rospkg catkin_pkg
    ```
    
2. Necessary tricks to use python 3 with ros

    * This project is based on python3, however ROS has not changed to python3. When the next ROS distribution will be out on python3, this sequence will become obsolete.

    * Because of issues with python3 and ROS, the project uses a workspace dedicated to python3. 

    ```bash
    mkdir catkin_ws_py3
    cd catkin_ws_py3
    mkdir src
    cd src
    catkin_init_workspace
    cd ..
    wstool init 
    wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5 
    wstool up 
    cd src
    git clone -b melodic https://github.com/ros-perception/vision_opencv.git 
    cd ..
    catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3  -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so 
    source devel/setup.bash
    ```
3. Clone the repo

    ```bash
    cd catkin_ws_py3/src
    git clone https://github.com/introlab/dotmask.git
    cd ..
    catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3  -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so 
    source devel/setup.bash
    ```
    
4. Download the NN
    * You need to download the desired neural network from their original repo.

    YOLACT and YOLACT++: 
    ```bash
    cd catkin_ws_py3/src/dotmask/nn
    git clone https://github.com/dbolya/yolact.git
    ```
      * For YOLACT++ you need to compile DCNv2, this may require elevated permissions
        ```bash
        cd catkin_ws_py3/src/dotmask/nn/external/DCNv2
        python3 setup.py build develop
        ```

    Mask R-CNN: 
    ```bash
    cd catkin_ws_py3/src/dotmask/nn
    git clone https://github.com/matterport/Mask_RCNN.git
    ```

4. Download the NN weigths

    Download the weigths and place them in the "catkin_ws_py3/src/dotmask/weights" folder of this repo.
    * Those links are from the orignal repos

    YOLACT:
    * https://drive.google.com/uc?id=1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0&export=download 

    YOLACT++:
    * https://drive.google.com/uc?id=1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP&export=download

    Mask R-CNN:
    * https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5 

## Benchmarking on TUM

1. Start RTABMap-ros
    ```bash
    roslaunch dotmask dotmask-tum.launch
    ```

2. Start DOTMask 
    * With YOLACT
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=yolact --input=tum
    ```
    * With YOLACT++ 
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=yolact++ --input=tum
    ```
    * With Mask R-CNN 
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=mrcnn --input=tum
    ```
    
3. Start a TUM rosbag
    * To run the rosbag with rtabmap, make sure to do the following steps for the desired sequence. 
    ```bash
    wget http://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.bag
    rosbag decompress rgbd_dataset_freiburg3_walking_static.bag
    wget https://gist.githubusercontent.com/matlabbe/897b775c38836ed8069a1397485ab024/raw/6287ce3def8231945326efead0c8a7730bf6a3d5/tum_rename_world_kinect_frame.py
    python tum_rename_world_kinect_frame.py rgbd_dataset_freiburg3_walking_static.bag
    ```
    
    * Run the rosbag
    ```bash
    rosbag play --clock rgbd_dataset_freiburg3_walking_static.bag -r 0.1
    ```


## Run DOTMask on xtion
   * Opeeni2 is required to run the camera, install ros-melodic-openni2-launch with apt

1. Start RTABMap-ros
    ```bash
    roslaunch dotmask dotmask-xtion.launch
    ```
    
2. Start the xtion
    ```bash
    roslaunch openni2_launch openni2.launch depth_registration:=true
    ```

3. Start DOTMask
    * If you want to use YOLACT++ or Mask-RCNN, change the --nn field for "yolact++" or "mrcnn" respectively
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=yolact --input=xtion
    ```
