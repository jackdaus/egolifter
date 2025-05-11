# About

This project is a fork of [EgoLifter](https://github.com/facebookresearch/egolifter). It contains some additional experiments. 

# Setup 

## Install dependencies 

First, clone this repo. 

```bash
git clone https://github.com/jackdaus/egolifter.git

cd egolifter
```


We are using [uv](https://docs.astral.sh/uv) to manage packages. Make sure you have it installed. Then, to install our dependencies, run,

```bash
uv sync
```

## Set up SAM and GroundingDINO

Download SAM and GroundingDINO model weights.

```bash
# SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# GroundingDINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM 2.1
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

wget $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; }
wget $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url";}
wget $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; }
wget $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; }
```

Set up environment variables. First, make a copy of the bash script.

```bash
cp setup_env.bash.template setup_env.bash
```

Change the values in `setup_env.bash` as needed. You will probably need to edit the variable for `EGOLIFTER_PATH`.

Then run,

```bash
source setup_env.bash
```

# Download Data

First, access ADT through [this link](https://www.projectaria.com/datasets/adt/#download-dataset) and download the `ADT_download_urls.json` file. That file contains the download links for the dataset.

Next, prepare a directory where you want to save the downloaded and processed dataset as follows. The below code then copies this file into a directory that will hold the downloaded dataset (the `ADT_DATA_ROOT` directory).

```bash
# Change the following to directories where you want to save the dataset
export ADT_DATA_ROOT=${HOME}/cs-747-project/adt
export ADT_PROCESSED_ROOT=${HOME}/cs-747-project/adt_processed

mkdir -p $ADT_DATA_ROOT
mkdir -p $ADT_PROCESSED_ROOT

# Edit this path appropriately
cp /path/to/ADT_download_urls.json $ADT_DATA_ROOT
```

Then run the following script to download and process the dataset.

```bash
uv run bash scripts/download_process_adt.bash
```

# Training

## Log into wandb (optional)

You can log into wandb to visualize your training progress in real time. 

```bash
uvx wandb login
```

## Train a vanilla 3DGS model 

First, we will train a basic vanilla 3DGS model. 

Make sure your environment variables are set. 

```bash
source setup_env.bash
ADT_PROCESSED_ROOT=${HOME}/cs-747-project/adt_processed
OUT_PATH=${HOME}/cs-747-project/output/adt
SCENE_NAME=Apartment_release_golden_skeleton_seq100_10s_sample
```

Run the training.

```bash
uv run python train_lightning.py \
    scene.scene_name=${SCENE_NAME} \
    scene.data_root=${ADT_PROCESSED_ROOT} \
    exp_name=3dgs \
    output_root=${OUT_PATH} \
    wandb.project=egolifter_adt
```

### Visualize the vanilla 3DGS results

This will start a local server on http://localhost:8080/. Open that link in a browser.
Tip: When you begin, click on "Nearest camera" to snap to a sensible viewing angle. 

```bash
uv run python viewer.py \
    ${OUT_PATH}/${SCENE_NAME}/vanilla_3dgs \
    --data_root ${ADT_PROCESSED_ROOT}  \
    --reorient disable 
```

## Train an Egolifter model

Next, we will try training the full EgoLifter setup. Set the environment variables as needed. This training takes a couple of hours.

```bash
source setup_env.bash
SCENE_NAME=Apartment_release_work_skeleton_seq131
ADT_PROCESSED_ROOT=${HOME}/cs-747-project/adt_processed
OUT_PATH=${HOME}/cs-747-project/output/adt

uv run python train_lightning.py \
	scene.scene_name=${SCENE_NAME} \
    scene.data_root=${ADT_PROCESSED_ROOT} \
    model=unc_2d_unet \
    model.unet_acti=sigmoid \
    model.dim_extra=16 \
    lift.use_contr=True \
    exp_name=egolifter \
    output_root=${OUT_PATH} \
    wandb.project=egolifter_adt
```

### Visualize the EgoLifter results

This will start a local server on http://localhost:8080/. Open that link in a browser.
Tip: When you begin, click on "Nearest camera" to snap to a sensible viewing angle. 

```bash
uv run python viewer.py \
    ${OUT_PATH}/${SCENE_NAME}/unc_2d_unet_egolifter \
    --data_root ${ADT_PROCESSED_ROOT}  \
    --reorient disable \
    --feat_pca
```

# Extra

## Run visualizer of an ADT dataset (optional)

For WSL, you can run the rerun server to enable viewing on windows. This will forward the app to localhost:9090. (To run this command, make sure the python venv is activated.)

```bash
rerun --serve-web
```

And then create a .rrd file from a sequence. This file can then be loaded into rerun.

```bash
uvx --from projectaria-tools viewer_projects_adt --sequence_path ./path/to/adt_sequence --rrd_output_path adt_dataset.rrd
```

for example, 

```bash
uvx --from projectaria-tools viewer_projects_adt --sequence_path ./adt/Apartment_release_multiskeleton_party_seq121_71292 --rrd_output_path Apartment_release_multiskeleton_party_seq121_71292.rrd
```

## Jupyter (optional)

To create a Jupyter kernel, run, 

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=egolifter
```