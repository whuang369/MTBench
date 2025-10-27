# How to run MTBench on UW Madison CHTC

## Prepare Isaacgym Package

First, download the isaac gym package from [official website](https://developer.nvidia.com/isaac-gym), then scp this file into your /staging:

```bash
scp IsaacGym_Preview_4_Package.tar.gz whuang369@ap2001.chtc.wisc.edu:/staging/YOUR_NETID
```
(replace YOUR_NETID with your netid)

## Interactive Job

To submit an interactive job, run
```bash
sh submit_gpu_i.sh MEMORY DISK GLOBAL_GPU_MEMORY
```
inside `chtc/`. Replace MEMORY, DISK, and GLOBAL_GPU_MEMORY with your requested memory(GB), disk size(GB), and global gpu memory(MB).

After entering your interactive job, run the following commands in sequence to install necessary packages:

```bash
git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout dynamic_task_counts
source install.sh
wandb login YOUR_WANDB_API_KEY
```

Then, try executing job scripts like

```bash
sh exec/ppo_exps/mt10-rand/famo.sh
```

If there's no error message and the result is synced on wandb.ai, then everything is good to go!

## Submit Job

### Create Job Scripts

To submit a complete job on chtc, first create a job script inside `chtc/commands` directory. An example is
```bash
#!/bin/bash
git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout mask_on_successful_tasks
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
source miniconda3/bin/activate
conda create -y -n py38env python=3.8
conda activate py38env
find $CONDA_PREFIX -name "libpython3.8.so.1.0"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda create -y -n mtbench python=3.8
conda activate mtbench
pip install -e .
pip install skrl
pip install moviepy
pip install numpy==1.23.5
cp /staging/whuang369/IsaacGym_Preview_4_Package.tar.gz .
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
cd ../..
wandb login YOUR_WANDB_API_KEY

sh exec/ppo_exps/mt10-rand/famo.sh
```
Everything before `sh exec/ppo_exps/mt10-rand/famo.sh` is installing necessary packages and settings, so please always keep this part in the front of your job script.

Then, replace `sh exec/ppo_exps/mt10-rand/famo.sh` either with a single script already in the repo, or with any commands that you wanna run.

### Submit Your Job Script

After job script is properly saved in `chtc/commands` , go back to `chtc/` and run

```bash
sh submit.sh JOB_SCRIPT_NAME MEMORY DISK GLOBAL_GPU_MEMORY JOB_LENGTH
```
Replace JOB_SCRIPT_NAME, MEMORY, DISK, GLOBAL_GPU_MEMORY, JOB_LENGTH with your job script name, requested memory(GB), requested disk size(GB), requested global gpu memory(MB), and your job length.

### Recommanded Settings

```markdown
MEMROY = 100
DISK = 100
GLOBAL_GPU_MEMORY = 10240
JOB_LENGTH = medium
```
The setting above is enough for reproducing mt10-rand and mt50-rand ppo famo.