# lect

location embedded contrastive training

## Data

- https://doi.org/10.11588/data/IUTCDN

## Build

1. build docker image from src
    ```
    docker build -t lect .
    ```

2. initialize a container
    ```
    docker run -it --gpus device=0 --name lect --mount type=bind,source="$(pwd)",target=/app lect
    ```
