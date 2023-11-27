#!/bin/sh

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Ensure the -v volume mount line continues without line breaks
docker run --runtime=nvidia --privileged --rm -it \
           --volume=$XSOCK:$XSOCK:rw \
           --volume=$XAUTH:$XAUTH:rw \
           --volume=/home/mukul/Documents/Mobile_Robotics/carla_ros_bridge/ros-bridge:/home/carla_melodic/catkin_ws/src/ee585_carla_project/ros-bridge \
           --volume=$HOME:$HOME \
           --shm-size=1gb \
           --env="XAUTHORITY=${XAUTH}" \
           --env="DISPLAY=${DISPLAY}" \
           --env=TERM=xterm-256color \
           --env=QT_X11_NO_MITSHM=1 \
           --net=host \
           -u "carla_melodic"  \
           carla:0.9.10 \
           bash
