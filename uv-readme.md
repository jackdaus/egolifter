# About

This project is using the `uv` package manager. Install it by following [these instructions](https://docs.astral.sh/uv).

# Install dependencies 

```bash
uv sync
```

# Log into wandb

```bash
uvx wandb login
```

# Download Models

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

# Set up Grounded-Segment-Anything

```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

cd Grounded-Segment-Anything/

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

uv pip install -e ./segment_anything

uv pip install -e ./GroundingDINO

uv pip install --upgrade "diffusers[torch]"
```

# To run the pipeline on the ADT data

Set up environment variables.

```bash
source setup_env.bash
```

## Download Data

First access ADT through this link and download the ADT_download_urls.json file, which contains the download links for the dataset.

Then prepare a directory where you want to save the downloaded and processed dataset as follows. And put the ADT_download_urls.json in the $ADT_DATA_ROOT directory.

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

# Run visualizer of an ADT dataset (TODO make work with WSL)

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

TODO: figure out how to stream data to rerun... not working for me in wsl.

# Jupyter

To create a Jupyter kernel, run, 

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=egolifter
```