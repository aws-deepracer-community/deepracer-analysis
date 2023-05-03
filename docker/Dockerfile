FROM jupyter/minimal-notebook:python-3.7

COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    conda install -y geos
RUN jupyter lab build
