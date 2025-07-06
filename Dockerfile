FROM timeloopaccelergy/timeloop-accelergy-pytorch:latest-arm64

ENV USER_UID=501
ENV USER_GID=20

ENV JUPYTER_SWITCHES="--NotebookApp.token=''"

RUN pip install /usr/local/src/timeloopfe/

WORKDIR /opt/project/