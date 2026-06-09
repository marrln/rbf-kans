#!/bin/bash

########################################
# Configuration arguments
########################################
DATASET="cifar10"
TEST_VERSION=""
SEED=42
WITH_LOGITS=1
RESIZE=""
LAYERS="36"
NUM_GRIDS="4"
GRID_MIN="-1.5"
GRID_MAX="1.5"
SCALE="5"
MODE="RSWAFF"
RESIDUAL=0
DYNAMIC=0
USE_V2=0
NO_NORMALIZE=0
NO_NORMALIZE_RBF=0
DROPOUT=0.1
LINEAR_DROPOUT=0.1
EPOCHS=2
PATIENCE=50
BATCH=100
LR=5e-2
LR_FACTOR=0.5
LR_PATIENCE=10
OPTIMIZER="AdamW"
WEIGHT_DECAY=5e-2
MOMENTUM=0.9
PYTHON=python

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
########################################
# Internal variables
########################################
dryrun=0
verbose=0
exp_hash=

dry_run() {
    if [ $dryrun -lt 1 ]; then
        "$@"
    fi
}

print_verbose() {
    if [ $verbose -ge 1 ]; then
        echo "$*"
    fi
}

print_exec() {
    print_verbose "[EXEC] $*"
    dry_run "$@"
}

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
    -h, --help              Show this help
    -s, --seed SEED         Set random seed
    -d, --dry-run           Dry run (no execution)
    -v, --verbose           Verbose output
    --dataset NAME          Dataset name (cifar10, cifar100)
    --residual              Enable residual connections
    --dynamic               Enable dynamic RBF
    --use-v2                Use V2 RBFKAN layers
    --no-normalize          Disable layer normalization
    --no-normalize-rbf      Disable RBF normalization
    --hash HASH             Use existing config hash
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -d|--dry-run) dryrun=1; verbose=1; shift ;;
        -v|--verbose) verbose=1; shift ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --residual) RESIDUAL=1; shift ;;
        --dynamic) DYNAMIC=1; shift ;;
        --use-v2) USE_V2=1; shift ;;
        --no-normalize) NO_NORMALIZE=1; shift ;;
        --no-normalize-rbf) NO_NORMALIZE_RBF=1; shift ;;
        --hash) exp_hash="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

THIS_DIR="$(dirname "$(realpath "$0")")"
cd "$THIS_DIR" || exit 1

if [ -z "$exp_hash" ]; then
    CONFIGS=()
    [ -n "$LAYERS" ] && CONFIGS+=(--layers "$LAYERS")
    [ -n "$NUM_GRIDS" ] && CONFIGS+=(--num-grids "$NUM_GRIDS")
    [ -n "$GRID_MIN" ] && CONFIGS+=(--grid-min "$GRID_MIN")
    [ -n "$GRID_MAX" ] && CONFIGS+=(--grid-max "$GRID_MAX")
    [ -n "$SCALE" ] && CONFIGS+=(--scale "$SCALE")
    [ -n "$MODE" ] && CONFIGS+=(--mode "$MODE")
    [ "$RESIDUAL" -eq 1 ] && CONFIGS+=(--residual)
    [ "$DYNAMIC" -eq 1 ] && CONFIGS+=(--dynamic)
    [ "$USE_V2" -eq 1 ] && CONFIGS+=(--use-v2)
    [ "$NO_NORMALIZE" -eq 1 ] && CONFIGS+=(--no-normalize)
    [ "$NO_NORMALIZE_RBF" -eq 1 ] && CONFIGS+=(--no-normalize-rbf)
    [ -n "$RESIZE" ] && CONFIGS+=(--resize $RESIZE)
    [ -n "$DROPOUT" ] && CONFIGS+=(--dropout "$DROPOUT")
    [ -n "$LINEAR_DROPOUT" ] && CONFIGS+=(--dropout-linear "$LINEAR_DROPOUT")
    [ -n "$EPOCHS" ] && CONFIGS+=(--epochs "$EPOCHS")
    [ -n "$PATIENCE" ] && CONFIGS+=(--patience "$PATIENCE")
    [ -n "$BATCH" ] && CONFIGS+=(--batch "$BATCH")
    [ -n "$LR" ] && CONFIGS+=(--lr "$LR")
    [ -n "$LR_FACTOR" ] && CONFIGS+=(--lr-factor "$LR_FACTOR")
    [ -n "$LR_PATIENCE" ] && CONFIGS+=(--lr-patience "$LR_PATIENCE")
    [ -n "$OPTIMIZER" ] && CONFIGS+=(--optimizer "$OPTIMIZER")
    [ -n "$WEIGHT_DECAY" ] && CONFIGS+=(--weight-decay "$WEIGHT_DECAY")
    [ -n "$MOMENTUM" ] && CONFIGS+=(--momentum "$MOMENTUM")
    [ -n "$SEED" ] && CONFIGS+=(--seed "$SEED")
    [ -n "$TEST_VERSION" ] && CONFIGS+=(--test-version "$TEST_VERSION")
    [ "$WITH_LOGITS" -eq 1 ] && CONFIGS+=(--with-logits)
    CONFIGS+=(--dataset "$DATASET")

    cmd=("$PYTHON" "$THIS_DIR/create_config.py" "${CONFIGS[@]}" --export --hash)
    print_verbose "[EXEC] ${cmd[*]}"
    exp_hash=$(dry_run "${cmd[@]}")
    if [ $dryrun -ge 1 ]; then
        exp_hash="test_hash"
    fi
fi

if [ -n "$exp_hash" ]; then
    print_verbose "[INFO] Configuration Hash: $exp_hash"
fi

set_test_version=()
[ -n "$TEST_VERSION" ] && set_test_version=(--test-version "$TEST_VERSION")

print_exec "$PYTHON" "$THIS_DIR/train_model.py" --dataset "$DATASET" --hash "$exp_hash" "${set_test_version[@]}"
print_exec "$PYTHON" "$THIS_DIR/test_model.py" --dataset "$DATASET" --hash "$exp_hash" "${set_test_version[@]}" --epoch best
print_exec "$PYTHON" "$THIS_DIR/extract_rslt_stats.py" --dataset "$DATASET" --hash "$exp_hash" "${set_test_version[@]}" --epoch best