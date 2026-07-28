#!/bin/bash
set -e

MODE="${1:-}"
shift || true

case "$MODE" in
  predict)
    exec python3 prediction_finetuned_tabpfn.py "$@"
    ;;
  embed)
    exec python3 generate_embedding.py "$@"
    ;;
  train)
    exec python3 training_finetuned_tabpfn.py "$@"
    ;;
  train-nested)
    exec python3 training_finetuned_tabpfn_Nested.py "$@"
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    cat <<'EOF'
VALANCE - HIV-1 broadly neutralizing antibody resistance prediction

Usage:
  docker run ... zzshenglab/valance:v1.0 <mode> [options]

Modes:
  predict       Predict resistance for new sequences using a pretrained model
  embed         Generate ESM2 embeddings for training data
  train         Train a VALANCE model (single train/test split)
  train-nested  Train a VALANCE model with nested cross-validation
  bash          Drop into an interactive shell inside the image

Run with a mode and --help to see that script's options, e.g.:
  docker run --rm zzshenglab/valance:v1.0 predict --help
EOF
    exit 0
    ;;
esac
