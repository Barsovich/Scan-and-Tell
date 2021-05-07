# Scan-and-Tell [[Report](https://github.com/Barsovich/Scan-and-Tell/blob/main/report.pdf)]
This is the repository for the [Scan and Tell](https://github.com/Barsovich/Scan-and-Tell/blob/main/report.pdf) project. The goal of the project is to improve the state-of-the-art 3D Dense Captioning architecture using sparse convolutions.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.2.0
* CUDA 10.0

### Virtual Environment
```
conda create -n scan-and-tell python==3.7
source activate scan-and-tell
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

### Install Backbones

(1) Clone the Scan-and-Tell repository.
```
git clone --recurse-submodules https://github.com/Barsovich/Scan-and-Tell.git
cd Scan-and-Tell
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** We further modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.

(5) Compile PointNet2

```
cd lib/pointnet2
python setup.py install
```

(6) Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `config/config_votenet.py`.
## Data Preparation

For downloading the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet Instructions](data/scannet/README.md) for more details.

Download the dataset by simply executing the wget command:
```
wget <download_link>
```

1. Download the ScanRefer dataset and unzip it under `data/`.
   
    a) Run `scripts/organize_scanrefer.py`
2. Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Depending on the backbone you would like to use, set the model to `votenet` or `pointgroup`. You can also do both.
```
cd data/scannet/
python batch_load_scannet_data.py --model <model>
```
5. Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and decompress [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip).

    c. Change the data paths in `config.py` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    python script/compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds; you need ~36GB to store the generated HDF5 database:
    ```shell
    python script/project_multiview_features.py --maxpool
    ```
    > You can check if the projections make sense by projecting the semantic labels from image to the target point cloud by:
    > ```shell
    > python script/project_multiview_labels.py --scene_id scene0000_00 --maxpool
    > ```
## Usage
### Training
#### PointGroup backbone
1. (Optional) Configure the desired settings in `config/pointgroup_run1_scannet.yaml`
2. Run the training script
```shell
python train_pointgroup.py --config config/pointgroup_run1_scannet.yaml
```
#### VoteNet backbone
1. (Optional) Configure the desired settings in `config/votenet_args.yaml`
2. Run the training script
```shell
python train_votenet.py --config config/votenet_args.yaml
```
### Evaluation
#### PointGroup backbone
1. (Optional) Configure the desired settings in `config/pointgroup_run1_scannet.yaml`
2. Run the training script
```shell
python eval_pointgroup.py --config config/pointgroup_run1_scannet.yaml
```
If you use the same configuration file as the config parameter, it will automatically resume the training from the last saved checkpoint.
#### VoteNet backbone
1. (Optional) Configure the desired settings in `config/votenet_eval_args.yaml`
2. Run the training script
```shell
python eval_votenet.py --config config/votenet_eval_args.yaml
```

## Branches
The `main` should be used for all steps. The `rg` branch is currently experimental and aims to utilize a graph message passing network to further improve the results.

## Acknowledgement
This repository uses the PointGroup implementation from https://github.com/Jia-Research-Lab/PointGroup and the VoteNet implementation from https://github.com/daveredrum/ScanRefer. We would like to thank Dave Z. Chen and Jia-Research-Lab for their implementations.
