#!/bin/bash
git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout chtc
sh install_miniconda.sh
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
sh exec/ppo_exps/mt10-rand/famo.sh
