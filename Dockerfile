## Dockerfile

ARG BASE_IMAGE=pytorch/pytorch:latest
FROM ${BASE_IMAGE} as dev-base


SHELL ["/bin/bash", "--login","-o","pipefail", "-c"]
RUN conda init bash

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda create --name sitaware
SHELL ["conda", "run", "-n", "sitaware", "/bin/bash", "-c"]

#Potentially change the location of this
WORKDIR /root/

ARG GITHUB_TOKEN
RUN apt update 
RUN apt -y install git
RUN git clone https://max-kaufmann:${GITHUB_TOKEN}@github.com/AsaCooperStickland/situational-awareness.git 
RUN pip install -e /root/situational-awareness 
RUN pip install tiktoken 
RUN pip install debugpy