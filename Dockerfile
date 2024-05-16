FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# system dependencies
RUN apt-get update &&\
    apt-get install -y \
    git \
    libtiff5

# a new conda solver libmamba
RUN conda update -n base conda &&\
    conda install -n base -y conda-libmamba-solver &&\
    conda config --set solver libmamba

# conda install packages
RUN conda install -c conda-forge -y \
    rasterio \
    imageio \
    geojson \
    matplotlib

WORKDIR /app