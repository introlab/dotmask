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
    catkin_init_ws
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
