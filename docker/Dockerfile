FROM jupyter/minimal-notebook:python-3.7

COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
RUN jupyter lab build
USER root
RUN apt-get update && apt-get install -qq libgl1 libgeos-dev libgeos-c1v5
