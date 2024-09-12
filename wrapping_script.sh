#!/bin/bash -l
# ^ Shebang-Line, so that it is known that this is a bash file
# -l means 'load this as login shell', so that /etc/profile gets loaded and you can use 'module load' or 'ml' as usual

# If you don't use this script via `./run.sh' or just `srun run.sh', but like `srun bash run.sh', please add the '-l' there too.
# Like this:
# srun bash -l run.sh

# Load modules your program needs, always specify versions!
#module load release/23.04 GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.7.1
#source </path/to/workspace/python-environments/torchvision_env>/bin/activate #activate virtual environment

# Load your script. $@ is all the parameters that are given to this shell file.
python3 mnistFashion.py $@
