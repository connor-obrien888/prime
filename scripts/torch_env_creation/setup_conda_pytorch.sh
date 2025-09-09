#!/bin/sh
echo 'Please select the version of PyTorch to install:'
nl pt_conda.list
count="$(wc -l pt_conda.list | cut -f 1 -d' ')"
n=""
while true; do
    read -p 'Select option: ' n
    # If $n is an integer between one and $count...
    if [ "$n" -eq "$n" ] && [ "$n" -gt 0 ] && [ "$n" -le "$count" ]; then
        break
    fi
done
value=( $(sed -n "${n}p" pt_conda.list) )
echo "PyTorch $value will now be installed in a GPU enabled conda environment."

PT_VER=${value[0]}
CUDA_VER=${value[2]}
VISION_VER=${value[4]}
AUDIO_VER=${value[6]}
PY_VER=${value[8]}

cat << EOF > pt${PT_VER//./}_env.yaml
name: pt${PT_VER//./}gpu_conda
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=${PY_VER}.*
  - libblas=*=*mkl
  - numpy
  - scipy
  - pandas
  - hdf5=*=*openmpi*
  - h5py=*=*openmpi*
  - pytorch=${PT_VER}
  - torchvision=${VISION_VER}
  - torchaudio=${AUDIO_VER}
  - pytorch-cuda=${CUDA_VER}
EOF

module load conda

CONDA_OVERRIDE_CUDA=$CUDA_VER mamba env create -f pt${PT_VER//./}_env.yaml

conda activate pt${PT_VER//./}gpu_conda

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cp activate_env_vars.sh $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
cp deactivate_env_vars.sh $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda deactivate