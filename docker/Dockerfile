FROM jupyter/minimal-notebook:python-3.9

COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
RUN jupyter lab build
