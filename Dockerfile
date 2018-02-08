FROM tensorflow/tensorflow:1.5-gpu-py3

MAINTAINER artex.xh@gmail.com

WORKDIR /

ENV PYTHONPATH $PYTHONPATH:/
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get -y update && \
    apt-get -y install awscli cowsay nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install dask[complete]

ADD . /

ENTRYPOINT python ./app.py $ARGUMENTS