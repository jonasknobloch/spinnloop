FROM timeloopaccelergy/timeloop-accelergy-pytorch:latest-arm64

ENV USER_UID=501
ENV USER_GID=20

ENV JUPYTER_SWITCHES="--NotebookApp.token=''"
ENV HF_HOME=/opt/project/hf

RUN pip install /usr/local/src/timeloopfe/

WORKDIR /opt/project/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .