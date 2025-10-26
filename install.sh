export SCRATCH_DIR=$(pwd)

export PIP_CACHE_DIR=$SCRATCH_DIR/pip_cache
export TORCH_EXTENSIONS_DIR=$SCRATCH_DIR/torch_extensions
export HF_HOME=$SCRATCH_DIR/huggingface
export TMPDIR=$SCRATCH_DIR/tmp
export TRANSFORMERS_CACHE=$SCRATCH_DIR/transformers
mkdir -p $PIP_CACHE_DIR $TORCH_EXTENSIONS_DIR $HF_HOME $TMPDIR $TRANSFORMERS_CACHE

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