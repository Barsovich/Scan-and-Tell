#!/bin/bash
# Script for setting up a Colab virtual machine from scratch

# Assumptions: 
# - CUDA 10.1 is already installed which is the current state of Colab

# Get the absolute path to the directory containing this script. Source: https://stackoverflow.com/a/246128.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Include common functions and constants.
source "${SCRIPT_DIR}"/common.sh

USAGE='Usage: bash provision.sh [github-credential]'

# Hack to make the script ask for sudo password before printing any steps.
sudo printf ''

print_green 'Installing required packages'

print_green 'Update the apt package index'
sudo apt-get --assume-yes update
exit_if_failed 'Updating the apt package index failed.'

print_green 'Install git'
sudo apt-get --assume-yes install git
exit_if_failed 'Installing git failed.'

print_green 'Clone the Scan and Tell repository, you might be asked for GitHub credentials.'
git clone https://github.com/Barsovich/Scan-and-Tell.git
exit_if_failed 'Cloning the GitHub repository failed.'

# This step of the installer follows the tutorial at https://towardsdatascience.com/conda-google-colab-75f7c867a522
print_green 'Install conda'
## unset PYTHONPATH # This is required since it may cause problems during the installation.
MINICONDA_INSTALLER_SCRIPT=Miniconda3-py37_4.9.2-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
exit_if_failed 'Downloading conda failed.'

chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
exit_if_failed 'Installing conda failed.'

print_green 'Update conda'
conda install --channel defaults conda python=3.7.0 --yes
conda update --channel defaults --all --yes
exit_if_failed 'Updating conda failed'

print_green 'Final conda and python versions:'
conda --version
python --version

# TODO: Make sure that the following command adds the directory to the
#       path. We might need to call 'source ~/.bashrc' or similar.
print_green 'Add conda module directory to the path'
if ['$PYTHONPATH' == '']
then
    export PYTHONPATH="/usr/local/bin/python"
else
    export PYTHONPATH="$PYTHONPATH:/usr/local/bin/python"
fi

print_green 'Create and source virtual environment'
conda create -n scan-and-tell python==3.7.0 --yes
source activate scan-and-tell

print_green '--------- Install PointGroup ---------'
print_green 'Clone PointGroup repository'
# POINTGROUPDIR = '$SCRIPT_DIR/PointGroup'
git clone https://github.com/llijiang/PointGroup.git --recursive 
exit_if_failed 'Cloning PointGroup repository failed.'

print_green 'Install required modules'

# Change requirement to pytorch 1.2
sed -i 's/torch==1.1/torch==1.2/' PointGroup/requirements.txt

pip install -r PointGroup/requirements.txt
exit_if_failed 'Installing requirements.txt failed.'
conda install -c bioconda google-sparsehash --yes
exit_if_failed 'Installing google-sparsehash failed.'

print_green 'Compile sparseconv'
conda install libboost --yes
exit_if_failed 'Installing libboost failed.'
conda install gcc_linux-64 --yes # conda install -c daleydeng gcc-5 fails.
exit_if_failed 'Installing gcc-5 failed.'
sed -i '5i\\ninclude_directories(/usr/include/boost)\n' PointGroup/lib/spconv/CMakeLists.txt
exit_if_failed 'Adding boost to the CMakeLists.txt failed.'
cd PointGroup/lib/spconv
python setup.py bdist_wheel
cd dist
pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl

print_green 'Compile pointgroup_ops'
cd ../../pointgroup_ops/
conda install gxx_linux-64 --yes
conda install cudatoolkit=10.0 --yes
python setup.py build_ext --include-dirs=/usr/local/cuda-10.0/targets/x86_64-linux/include/:/usr/local/envs/scan-and-tell/include/
python setup.py develop
exit_if_failed 'Failed to compile pointgroup_ops'

print_green 'Installing PointNet++'
cd ~/Scan-and-Tell/lib/pointnet2
python setup.py build_ext --include-dirs=/usr/local/cuda-10.0/targets/x86_64-linux/include/
python setup.py install

# Done!
print_done_message 'Setup succeeded.'