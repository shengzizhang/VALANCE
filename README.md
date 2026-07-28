# VALANCE

**V**iral **A**ntibody **L**anguage-model **A**ssisted **N**eutralization & **C**linical **E**valuation

VALANCE is a machine learning pipeline that predicts HIV-1 broadly neutralizing antibody (bNAb) resistance from Env sequences. It embeds sequences with ESM2, performs feature down-selection, and classifies neutralization sensitivity/resistance with a fine-tuned TabPFN transformer at a chosen IC₅₀ threshold (0.2, 1, or 50 µg/mL).

A docker image of the pipeline can be accessed via: [https://hub.docker.com/r/zzshenglab/nanohivseq](https://hub.docker.com/r/zzshenglab/valance)

## Requirements

* NVIDIA GPU with CUDA 12.6-compatible driver
* [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) / NVIDIA Container Toolkit installed

## Quick Start

### Predict resistance for new sequences

Given a FASTA file of new Env sequences, a pretrained model, and the antibody's reference alignment, VALANCE predicts resistance/sensitivity at the model's trained IC₅₀ threshold.

```bash
docker run -it --rm \
  --gpus all \
  -v "$(pwd)/models":/work/models \
  -v "$(pwd)/reference_alignment":/work/reference \
  -v "$(pwd)/input":/work/input \
  -v "$(pwd)/output":/work/output \
  zzshenglab/valance:v1.0 \
  predict \
  --Ab VRC01 \
  --model /work/models/VRC01_50.0_final_model_finetuned_tabpfn.joblib \
  --reference /work/reference/VRC01_reference_aln.fasta \
  --file /work/input/new_strains.fasta \
  --outfile /work/output/VRC01_prediction_thres50.0.txt
```

Pretrained models for all 25 bNAbs across all three thresholds are available for download at **[TBD]**. Reference alignments for each antibody are included in this repository under [`reference_alignment/`](https://github.com/shengzizhang/VALANCE/tree/main/reference_alignment) and baked into the image.

### Train a model from scratch

> **One-time setup:** Training uses TabPFN-2.5, whose model weights are gated under a non-commercial license. Before your first training run:
> 1. Visit [ux.priorlabs.ai](https://ux.priorlabs.ai) and log in (or create an account).
> 2. Go to the **Licenses** tab and accept the license for the TabPFN 2.5 model.
> 3. Go to **Account** and copy your API key.
> 4. Pass it to the container as an environment variable via `-e TABPFN_TOKEN="your-token-here"` on every `train`/`train-nested` run (see below).
>
> This token is personal and should **never** be baked into the image or committed to the repo — pass it at `docker run` time only, ideally via an `--env-file` you keep out of version control rather than typing it inline in a shared shell history.
>
> **Behind a corporate/institutional proxy?** TabPFN verifies your token over HTTPS on first use each run. If your network requires a proxy for outbound internet access, also pass `-e http_proxy=... -e https_proxy=...` alongside `-e TABPFN_TOKEN=...` — otherwise verification will silently retry and time out on every cross-validation fold, making training much slower than it needs to be (it still runs, just wastes time waiting).

Training is a two-step process: generate ESM2 embeddings for your training data, then fit the TabPFN classifier at a chosen IC₅₀ cutoff.

```bash
# 1. Generate ESM2 embeddings
docker run -it --rm \
  --gpus all \
  -v "$(pwd)/training_data":/work/data \
  -w /work/data \
  zzshenglab/valance:v1.0 \
  embed \
  --Ab VRC01 \
  --training /work/data/env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt \
  --thread 40

# 2. Train
docker run -it --rm \
  --gpus all \
  -e TABPFN_TOKEN="your-token-here" \
  -v "$(pwd)/training_data":/work/data \
  -v "$(pwd)/models":/work/models \
  -w /work/data \
  zzshenglab/valance:v1.0 \
  train \
  --Ab VRC01 \
  --cutoff 50.0 \
  --traindata /work/data/env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt \
  --train_embedding /work/data/VRC01_training.h5
```

For nested cross-validation results (reported in the manuscript), use `train-nested` in place of `train`:

```bash
docker run -it --rm \
  --gpus all \
  -e TABPFN_TOKEN="your-token-here" \
  -v "$(pwd)/training_data":/work/data \
  -v "$(pwd)/models":/work/models \
  -w /work/data \
  zzshenglab/valance:v1.0 \
  train-nested \
  --Ab VRC01 \
  --cutoff 50.0 \
  --traindata /work/data/env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt \
  --train_embedding /work/data/VRC01_training.h5
```

## Verified Test Run

The commands below are a minimal, confirmed-working test covering all three modes — useful for a small test run before training/testing on real antibodies & viruses.

```bash
# Predict
mkdir -p test_models test_reference test_input test_output
cp /path/to/VRC01_50.0_final_model_finetuned_tabpfn.joblib test_models/
cp /path/to/VRC01_reference_aln.fasta test_reference/
cp /path/to/your_test_strains.fasta test_input/

sudo docker run -it --rm \
  --gpus all \
  -v "$(pwd)/test_models":/work/models \
  -v "$(pwd)/test_reference":/work/reference \
  -v "$(pwd)/test_input":/work/input \
  -v "$(pwd)/test_output":/work/output \
  valance \
  predict \
  --Ab VRC01 \
  --model /work/models/VRC01_50.0_final_model_finetuned_tabpfn.joblib \
  --reference /work/reference/VRC01_reference_aln.fasta \
  --file /work/input/your_test_strains.fasta \
  --outfile /work/output/VRC01_test_prediction.txt
```

```bash
# Embed
mkdir -p test_training_data
# place a small training data table (same tab-delimited format as
# env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt)
# into test_training_data/ before running

sudo docker run -it --rm \
  --gpus all \
  -v "$(pwd)/test_training_data":/work/data \
  -w /work/data \
  valance \
  embed \
  --Ab VRC01 \
  --training /work/data/your_training_data.txt \
  --thread 4
```

```bash
# Train
sudo docker run -it --rm \
  --gpus all \
  -e TABPFN_TOKEN="your-token-here" \
  -e http_proxy=http://your-proxy:port \
  -e https_proxy=http://your-proxy:port \
  -e no_proxy=localhost,127.0.0.1 \
  -v "$(pwd)/test_training_data":/work/data \
  -v "$(pwd)/test_models_out":/work/models \
  -w /work/data \
  valance \
  train \
  --Ab VRC01 \
  --cutoff 1.0 \
  --traindata /work/data/your_training_data.txt \
  --train_embedding /work/data/VRC01_training.h5
```

Omit the `http_proxy`/`https_proxy`/`no_proxy` lines entirely if your network doesn't require a proxy for outbound internet access.


## Notes

* One image supports both training and prediction; the mode is selected by the first argument to `docker run` (`predict`, `embed`, `train`, `train-nested`, or `bash` for an interactive shell).
* Pretrained models, reference alignments used at inference time, and input/output data are bind-mounted rather than baked into the image, keeping the image reusable across antibodies and updatable as new models are released.
* Mount only the directories you need for a given mode — `-v host_path:container_path`.

## Citation

If you use VALANCE, please cite:

[TBD]
