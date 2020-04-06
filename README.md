# DOTMask

**D**ynamic **O**bject **T**racking and **Mask**ing is a simple and modular visual SLAM pipeline that to improves both localization and mapping in dynamic environments for mobile robotics. 

## Requirements

Python 3.6.9 (& 2.7), Cuda 10.2, PyTorch, ROS Melodic, RTAB-Map

*For better performances,install RTAB-Map with recomended/optional dependencies*

## Installation

1. Project dependencies

    ```bash
    sudo apt-get install python3-pip python3-yaml python3-catkin-pkg-modules python3-rospkg-modules python3-empy
    pip3 install torch torchvision cython
    pip3 install opencv-python pillow pycocotools matplotlib scikit-learn rospkg catkin_pkg
    ```
    
2. Necesseray tricks to use python3 with ros

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

## Benchmarking on TUM

1. Start RTABMap-ros
    ```bash
    roslaunch dotmask dotmask-tum.launch
    ```

2. Start DOTMask
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=yolact --input=tum
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
    ```bash
    source ~/catkin_ws_py3/devel/setup.bash
    cd ~/catkin_ws_py3/src/dotmask/src
    python3 dotmask_node.py --nn=yolact --input=xtion
    ```
