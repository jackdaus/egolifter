# About

This project is using the `uv` package manager. Install it by following [these instructions](https://docs.astral.sh/uv).

# Install dependencies 

```bash
uv sync
```

# Set up SAM and GroundingDINO

Download SAM and GroundingDINO models.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Set up environment variables. First, make a copy of the bash script.

```bash
cp setup_env.bash.template setup_env.bash
```

Change the values in `setup_env.bash` as needed. And then run,

```bash
source setup_env.bash
```

# Download Data

First, access ADT through [this link](https://www.projectaria.com/datasets/adt/#download-dataset) and download the ADT_download_urls.json file. That file contains the download links for the dataset.

Next, prepare a directory where you want to save the downloaded and processed dataset as follows. And put the ADT_download_urls.json in the $ADT_DATA_ROOT directory.

```bash
# Change the following to directories where you want to save the dataset
export ADT_DATA_ROOT=adt
export ADT_PROCESSED_ROOT=adt_processed

mkdir -p $ADT_DATA_ROOT
mkdir -p $ADT_PROCESSED_ROOT

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
```

```bash
SCENE_NAME=Apartment_release_golden_skeleton_seq100_10s_sample

uv run python train_lightning.py \
    scene.scene_name=${SCENE_NAME} \
    scene.data_root=${ADT_PROCESSED_ROOT} \
    exp_name=3dgs \
    output_root=./output/adt \
    wandb.project=egolifter_adt
```

### Visualize the vanilla 3DGS results

This will start a local server on http://localhost:8080/. Open that link in a browser.
Tip: When you begin, click on "Nearest camera" to snap to a sensible viewing angle. 

```bash
uv run python viewer.py \
    ./output/adt/${SCENE_NAME}/vanilla_3dgs \
    --data_root ${ADT_PROCESSED_ROOT}  \
    --reorient disable 
```

## Train an Egolifter model

Set the `OUT_PATH` variable to where you want to save the output of the training run.

```bash
OUT_PATH=${HOME}/cs-747-project/output/adt

uv run python train_lightning.py \
    scene.scene_name="Apartment_release_golden_skeleton_seq100_10s_sample" \
    scene.data_root=$ADT_PROCESSED_ROOT \
    model=unc_2d_unet \
    model.unet_acti=sigmoid \
    model.dim_extra=16 \
    lift.use_contr=True \
    exp_name=egolifter \
    output_root=$OUT_PATH \
    wandb.project=egolifter_adt
```

### Visualize the EgoLifter results

This will start a local server on http://localhost:8080/. Open that link in a browser.
Tip: When you begin, click on "Nearest camera" to snap to a sensible viewing angle. 

```bash
uv run python viewer.py \
    ./output/adt/${SCENE_NAME}/vanilla_3dgs \
    --data_root ${ADT_PROCESSED_ROOT}  \
    --reorient disable \
    --feat_pca
```

# Extra

## Run visualizer of an ADT dataset (Optional)

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

## Jupyter (Optional)

To create a Jupyter kernel, run, 

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=egolifter
```