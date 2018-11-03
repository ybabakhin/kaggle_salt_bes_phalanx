FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN pip --no-cache-dir install --upgrade \
        http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \
        torchvision==0.2.1 \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache

RUN apt-get -y update
RUN pip install --upgrade pip
## Install OpenCV
#RUN apt-get -y install git python3-setuptools
#RUN apt-get -y install cmake build-essential libjpeg-dev libpng-dev libgtk2.0-dev
#RUN apt-get -y install checkinstall pkg-config yasm gfortran git
#
#RUN apt-get -y install libjasper-dev
#RUN apt-get -y install libtiff5-dev
#
#RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
#RUN apt-get -y install libxine2-dev libv4l-dev
#RUN apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
#RUN apt-get -y install libqt4-dev libgtk2.0-dev libtbb-dev
#RUN apt-get -y install libatlas-base-dev
#RUN apt-get -y install libmp3lame-dev libtheora-dev
#RUN apt-get -y install libvorbis-dev libxvidcore-dev
#RUN apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev
#RUN apt-get -y install x264 v4l-utils
#
#RUN git clone https://github.com/opencv/opencv.git
#RUN cd opencv && mkdir build && cd build && cmake -D BUILD_EXAMPLES=OFF \
#-D BUILD_opencv_apps=OFF \
#-D BUILD_DOCS=OFF \
#-D BUILD_PERF_TESTS=OFF \
#-D BUILD_TESTS=OFF \
#-D CMAKE_INSTALL_PREFIX=/usr/local/ .. && make -j4 && make install && cd ../../ && rm -rf opencv

# Additinal Libraries
RUN pip install opencv-python==3.4.3.18
RUN apt-get -y install libsm6 libxext6 libxrender-dev
RUN pip install h5py==2.8.0
RUN pip install tqdm==4.24.0
RUN pip install albumentations==0.1.2
RUN pip install keras==2.2.0
# RUN pip install fastprogress==0.1.15

# RUN chmod +x entrypoint.sh

# User for writing

ARG UID=1005
ARG GID=1005

RUN groupadd -g ${GID} dev
RUN useradd -s /bin/bash -u $UID -g $GID p

USER p
