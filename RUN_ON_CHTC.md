# How to run MTBench on UW Madison CHTC Interactive GPU Job

## Prepare Isaacgym Package

First, install the isaac gym package from [official website](https://developer.nvidia.com/isaac-gym), then cd into the directory that `IsaacGym_Preview_4_Package.tar.gz` is saved, and scp this file into your /staging:

```bash
scp IsaacGym_Preview_4_Package.tar.gz whuang369@ap2001.chtc.wisc.edu:/staging/whuang369
```
(replace whuang369 with your netid)

## Clone this repo

After entering the interactive job, clone this repo into your job directory and go to the chtc branch:

```bash
git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout ppo
```

## Short Install

```bash
source install.sh
```

## Detailed installation guide in case you need to change something :-)

### Install Conda

First, install conda as follows:

```bash
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3

# Initialize Conda
source miniconda3/bin/activate
```

Then, run the following commands to install `libpython3.8.so.1.0`:

```bash
conda create -y -n py38env python=3.8
conda activate py38env
find $CONDA_PREFIX -name "libpython3.8.so.1.0"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Then, create the conda environment for our testing:

```bash
conda create -y -n mtbench python=3.8
conda activate mtbench
```

### Install This repo

Install necessary packages:
```bash
pip install -e .
pip install skrl
pip install moviepy
pip install numpy==1.23.5
```

### Install Isaac Gym

Install isaacgym as follows:
```bash
cp /staging/whuang369/IsaacGym_Preview_4_Package.tar.gz .
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
cd ../..
```
(replace whuang369 with your netid as well)

## Try yourself!

After all, you can test your program by creating `train.py` file, copy the code in "Basic Usage" into it, and try running the code as following:

```bash
python train.py task=meta-world-v2 checkpoint=runs/x/nn/x.pth
```