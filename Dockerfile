FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-dev libglib2.0-0 swig

RUN pip install --upgrade pip
RUN pip install matplotlib wandb tqdm gymnasium[all] hydra-core opencv-python