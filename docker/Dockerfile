FROM jupyter/minimal-notebook:latest

COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
RUN jupyter lab build

USER root
RUN apt-get update && apt-get upgrade -y && \ 
	apt-get install libgl1-mesa-glx -y

COPY ./visualization.py /opt/conda/lib/python3.9/site-packages/deepracer/model/visualization.py

