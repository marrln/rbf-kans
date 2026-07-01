#!/bin/bash

########################################
# Configuration arguments
########################################
TEST_VERSION=0     
SEED=42

# Dataset Configuration
DATASET="mnist"             # Options: cifar10, cifar100, mnist, ship_performance
RESIZE=""
AUGMENT_PROBABILITY=0.45

# Model Configuration
WITH_LOGITS=1
LAYERS="512 512"
RESIDUAL=0
NO_NORMALIZE=0
DROP_OUT=0.2
DYNAMIC_DROPOUT=0

# Training Configuration
EPOCHS=1000
BATCH=128
EARLY_STOPPING_PATIENCE=100
LR=3e-4
LR_FACTOR=0.5
LR_PATIENCE=25
OPTIMIZER="AdamW"   
GRAD_CLIP_LIMIT=1.0      
WEIGHT_DECAY=5e-4           
MOMENTUM=0.9        

# Python executable
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

# Helper function to split a space-separated string into an array
split_args() {
    local var="$1"
    local -n arr="$2"
    if [ -n "$var" ]; then
        # Use word splitting to separate values
        IFS=' ' read -r -a arr <<< "$var"
    else
        arr=()
    fi
}

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
    --dataset NAME          Dataset name (cifar10, cifar100, mnist, ship_performance)
    --residual              Enable residual connections
    --no-normalize          Disable layer normalization
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
        --no-normalize) NO_NORMALIZE=1; shift ;;
        --hash) exp_hash="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

THIS_DIR="$(dirname "$(realpath "$0")")"
cd "$THIS_DIR" || exit 1

if [ -z "$exp_hash" ]; then
    CONFIGS=()
    
    # Split multi-value arguments
    split_args "$LAYERS" layers_arr
    
    # Append them as separate elements
    [ ${#layers_arr[@]} -gt 0 ] && CONFIGS+=(--layers "${layers_arr[@]}")
    
    [ "$RESIDUAL" -eq 1 ] && CONFIGS+=(--residual)
    [ "$NO_NORMALIZE" -eq 1 ] && CONFIGS+=(--no-normalize)
    [ -n "$RESIZE" ] && CONFIGS+=(--resize "$RESIZE")
    [ -n "$AUGMENT_PROBABILITY" ] && CONFIGS+=(--augment-probability "$AUGMENT_PROBABILITY")
    [ -n "$DROP_OUT" ] && CONFIGS+=(--dropout "$DROP_OUT")
    [ "$DYNAMIC_DROPOUT" -eq 1 ] && CONFIGS+=(--dynamic-dropout)
    [ -n "$EPOCHS" ] && CONFIGS+=(--epochs "$EPOCHS")
    [ -n "$EARLY_STOPPING_PATIENCE" ] && CONFIGS+=(--patience "$EARLY_STOPPING_PATIENCE")
    [ -n "$BATCH" ] && CONFIGS+=(--batch "$BATCH")
    [ -n "$LR" ] && CONFIGS+=(--lr "$LR")
    [ -n "$LR_FACTOR" ] && CONFIGS+=(--lr-factor "$LR_FACTOR")
    [ -n "$LR_PATIENCE" ] && CONFIGS+=(--lr-patience "$LR_PATIENCE")
    [ -n "$OPTIMIZER" ] && CONFIGS+=(--optimizer "$OPTIMIZER")
    [ -n "$WEIGHT_DECAY" ] && CONFIGS+=(--weight-decay "$WEIGHT_DECAY")
    [ -n "$GRAD_CLIP_LIMIT" ] && CONFIGS+=(--clip-limit "$GRAD_CLIP_LIMIT")
    [ -n "$MOMENTUM" ] && CONFIGS+=(--momentum "$MOMENTUM")
    [ -n "$SEED" ] && CONFIGS+=(--seed "$SEED")
    [ -n "$TEST_VERSION" ] && CONFIGS+=(--test-version "$TEST_VERSION")
    [ "$WITH_LOGITS" -eq 1 ] && CONFIGS+=(--with-logits)
    CONFIGS+=(--dataset "$DATASET")

    cmd=("$PYTHON" "$THIS_DIR/create_configs_mlp.py" "${CONFIGS[@]}" --export --hash)
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