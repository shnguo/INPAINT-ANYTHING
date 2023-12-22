FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y tzdata ffmpeg libsm6 libxext6 \
        # libavfilter-dev \
        #   libavformat-dev \
        #   libavcodec-dev \
        #   libswresample-dev \
        #   libavutil-dev\
          wget \
          build-essential \
          git \
          yasm \
          ninja-build \
        #   cmake \
          git \
          vim \
        #   python3-pip \
        #   python-is-python3 \
        supervisor \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    # && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /workspace
COPY ./requirements.txt /workspace
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

