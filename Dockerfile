FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y libgl1-mesa-dev libglib2.0-0 swig libgl1-mesa-glx libosmesa6

RUN pip install --upgrade pip
RUN pip install matplotlib wandb tqdm gymnasium[all] hydra-core opencv-python dm_control