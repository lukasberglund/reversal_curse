ARG BASE_IMAGE=maxkaufmann1/runpod_owain:first
FROM ${BASE_IMAGE} as dev-base
# DELETE THE BELOW LINE
RUN rm -rf /root/situational-awareness/docker
ADD . /root/situational-awareness/docker
SHELL ["conda", "run", "-n", "sitaware", "/bin/bash", "-c"]
RUN python /root/situational-awareness/docker/user_setup.py
RUN apt install sudo
