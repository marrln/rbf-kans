#!/bin/bash

########################################
# Parameter Sweep Configuration
########################################

WITH_LOGITS=(1)
TEST_VERSIONS=("normalized")
SEEDS=(42)

LAYERS_LIST=("32 64 32" "128 64 32" "8 16")        # multi‑layer hidden sizes
NUM_GRIDS_LIST=("4 6" "8 12")                      # per‑layer grids
GRID_MIN_LIST=("-3 -1.5" "-2 -1")
GRID_MAX_LIST=("2 1.5" "3 2")
SCALE_LIST=("8 0.5" "10 1")
MODES=('RSWAFF')
RESIDUALS=(0)
DYNAMICS=(0)
USE_V2S=(0)
NO_NORMALIZES=(1)
NO_NORMALIZE_RBFS=(0)
DROPOUTS=(0.15)
DROPOUT_LINEAR_LIST=(0.1)

EPOCHS_LIST=(100)
PATIENCE_LIST=(50)
BATCH_SIZES=(16384)
LEARNING_RATES=(5e-2)
LR_FACTORS=(0.5)
LR_PATIENCE_LIST=(8)
OPTIMIZERS=("AdamW")
WEIGHT_DECAYS=(1e-4)
MOMENTUMS=("0.9")

########################################
# Runtime Configuration
########################################
MAX_PARALLEL_JOBS=1
DATASET="cifar100"
PYTHON=python
THIS_DIR=$(dirname "$(realpath "$0")")
RESULTS_DIR="$THIS_DIR/train/sweep_results"

########################################
########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
########################################

# ------------------------------------------------------------
# Generate all combinations using Python (tab‑separated)
# ------------------------------------------------------------
generate_combinations() {
    python3 -c '
import itertools, sys

# List of arrays – each array element can be a string with spaces
arrays = [
    '"${WITH_LOGITS[@]@Q}"',
    '"${TEST_VERSIONS[@]@Q}"',
    '"${SEEDS[@]@Q}"',
    '"${LAYERS_LIST[@]@Q}"',
    '"${NUM_GRIDS_LIST[@]@Q}"',
    '"${GRID_MIN_LIST[@]@Q}"',
    '"${GRID_MAX_LIST[@]@Q}"',
    '"${SCALE_LIST[@]@Q}"',
    '"${MODES[@]@Q}"',
    '"${RESIDUALS[@]@Q}"',
    '"${DYNAMICS[@]@Q}"',
    '"${USE_V2S[@]@Q}"',
    '"${NO_NORMALIZES[@]@Q}"',
    '"${NO_NORMALIZE_RBFS[@]@Q}"',
    '"${DROPOUTS[@]@Q}"',
    '"${DROPOUT_LINEAR_LIST[@]@Q}"',
    '"${EPOCHS_LIST[@]@Q}"',
    '"${PATIENCE_LIST[@]@Q}"',
    '"${BATCH_SIZES[@]@Q}"',
    '"${LEARNING_RATES[@]@Q}"',
    '"${LR_FACTORS[@]@Q}"',
    '"${LR_PATIENCE_LIST[@]@Q}"',
    '"${OPTIMIZERS[@]@Q}"',
    '"${WEIGHT_DECAYS[@]@Q}"',
    '"${MOMENTUMS[@]@Q}"',
]

for combo in itertools.product(*arrays):
    # Use TAB as delimiter to preserve spaces inside fields
    sys.stdout.write("\t".join(combo) + "\n")
' > /tmp/sweep_combos.txt
}

# ------------------------------------------------------------
# Parse command line arguments
# ------------------------------------------------------------
dryrun=0
verbose=0
purge=0
max_experiments=-1
no_pbar=0
assume_yes=
assume_no=

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: run-kan-sweep.sh [options]
    -d, --dry-run           Dry run
    -v, --verbose           Verbose
    -p, --purge             Purge existing outputs
    --max-experiments N     Limit total experiments
    -j, --jobs N            Parallel jobs
    --no-pbar               Disable progress bars
    -y / -N                 Assume yes / no
EOF
            exit 0 ;;
        -d|--dry-run) dryrun=1; verbose=1; shift ;;
        -v|--verbose) verbose=1; shift ;;
        -p|--purge) purge=1; shift ;;
        --max-experiments) max_experiments=$2; shift 2 ;;
        -j|--jobs) MAX_PARALLEL_JOBS=$2; shift 2 ;;
        --no-pbar) no_pbar=1; shift ;;
        -y) assume_yes=1; shift ;;
        -N) assume_no=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$THIS_DIR" || exit 1
mkdir -p "$RESULTS_DIR"

# Generate all combinations
generate_combinations
total_lines=$(wc -l < /tmp/sweep_combos.txt)
[ $max_experiments -gt 0 ] && [ $max_experiments -lt $total_lines ] && total_lines=$max_experiments

echo "========================================="
echo "KAN Parameter Sweep"
echo "Total experiments: $total_lines"
echo "Parallel jobs: $MAX_PARALLEL_JOBS"
echo "========================================="

if [ $dryrun -eq 0 ]; then
    if [[ -z $assume_no && -z $assume_yes ]]; then
        read -p "Proceed? (y/N): " -n 1 -r; echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    elif [[ $assume_no -eq 1 ]]; then
        exit 0
    fi
fi

# Log start
echo "Sweep started at $(date)" > "$RESULTS_DIR/sweep_log.txt"
echo "Total experiments: $total_lines" >> "$RESULTS_DIR/sweep_log.txt"

# ------------------------------------------------------------
# Main loop: read tab‑separated lines
# ------------------------------------------------------------
experiment_num=0
failed=0

while IFS= read -r line; do
    ((experiment_num++))
    [ $max_experiments -gt 0 ] && [ $experiment_num -gt $max_experiments ] && break

    # Split line into fields using tab delimiter
    IFS=$'\t' read -r -a fields <<< "$line"
    # field indices (must match order in product):
    # 0: with_logit, 1: test_version, 2: seed, 3: layers, 4: num_grids,
    # 5: grid_min, 6: grid_max, 7: scale, 8: mode, 9: residual,
    # 10: dynamic, 11: use_v2, 12: no_normalize, 13: no_normalize_rbf,
    # 14: dropout, 15: dropout_linear, 16: epochs, 17: patience,
    # 18: batch, 19: lr, 20: lr_factor, 21: lr_patience,
    # 22: optimizer, 23: weight_decay, 24: momentum

    # Build arguments for create_config.py – quote multi‑token values
    args=""
    [ "${fields[0]}" = "1" ] && args="$args --with-logits"
    [ -n "${fields[1]}" ] && args="$args --test-version \"${fields[1]}\""
    [ -n "${fields[2]}" ] && args="$args --seed ${fields[2]}"
    [ -n "${fields[3]}" ] && args="$args --layers \"${fields[3]}\""
    [ -n "${fields[4]}" ] && args="$args --num-grids \"${fields[4]}\""
    [ -n "${fields[5]}" ] && args="$args --grid-min \"${fields[5]}\""
    [ -n "${fields[6]}" ] && args="$args --grid-max \"${fields[6]}\""
    [ -n "${fields[7]}" ] && args="$args --scale \"${fields[7]}\""
    [ -n "${fields[8]}" ] && args="$args --mode ${fields[8]}"
    [ "${fields[9]}" = "1" ] && args="$args --residual"
    [ "${fields[10]}" = "1" ] && args="$args --dynamic"
    [ "${fields[11]}" = "1" ] && args="$args --use-v2"
    [ "${fields[12]}" = "1" ] && args="$args --no-normalize"
    [ "${fields[13]}" = "1" ] && args="$args --no-normalize-rbf"
    [ -n "${fields[14]}" ] && args="$args --dropout ${fields[14]}"
    [ -n "${fields[15]}" ] && args="$args --dropout-linear ${fields[15]}"
    [ -n "${fields[16]}" ] && args="$args --epochs ${fields[16]}"
    [ -n "${fields[17]}" ] && args="$args --patience ${fields[17]}"
    [ -n "${fields[18]}" ] && args="$args --batch ${fields[18]}"
    [ -n "${fields[19]}" ] && args="$args --lr ${fields[19]}"
    [ -n "${fields[20]}" ] && args="$args --lr-factor ${fields[20]}"
    [ -n "${fields[21]}" ] && args="$args --lr-patience ${fields[21]}"
    [ -n "${fields[22]}" ] && args="$args --optimizer ${fields[22]}"
    [ -n "${fields[23]}" ] && args="$args --weight-decay ${fields[23]}"
    [ -n "${fields[24]}" ] && args="$args --momentum ${fields[24]}"
    args="$args --dataset $DATASET"

    # Generate configuration hash
    cmd="$PYTHON $THIS_DIR/create_config.py $args --export --hash"
    [ $verbose -eq 1 ] && echo "[EXEC] $cmd"
    exp_hash=$(eval $cmd)
    [ $dryrun -eq 1 ] && exp_hash="dummy_$experiment_num"

    # Create hyperparameters.csv (if not dry run)
    if [ $dryrun -eq 0 ]; then
        test_version="${fields[1]}"
        [ -n "$test_version" ] && test_dir_name="test_${test_version}" || test_dir_name="test_0"
        config_dir="$THIS_DIR/train/${exp_hash}/${test_dir_name}/config"
        mkdir -p "$config_dir"
        {
            echo "parameter,value"
            echo "with_logits,${fields[0]}"
            echo "test_version,${fields[1]}"
            echo "seed,${fields[2]}"
            echo "layers,${fields[3]}"
            echo "num_grids,${fields[4]}"
            echo "grid_min,${fields[5]}"
            echo "grid_max,${fields[6]}"
            echo "scale,${fields[7]}"
            echo "mode,${fields[8]}"
            echo "residual,${fields[9]}"
            echo "dynamic,${fields[10]}"
            echo "use_v2,${fields[11]}"
            echo "no_normalize,${fields[12]}"
            echo "no_normalize_rbf,${fields[13]}"
            echo "dropout,${fields[14]}"
            echo "dropout_linear,${fields[15]}"
            echo "epochs,${fields[16]}"
            echo "patience,${fields[17]}"
            echo "batch_size,${fields[18]}"
            echo "learning_rate,${fields[19]}"
            echo "lr_factor,${fields[20]}"
            echo "lr_patience,${fields[21]}"
            echo "optimizer,${fields[22]}"
            echo "weight_decay,${fields[23]}"
            echo "momentum,${fields[24]}"
            echo "experiment_number,$experiment_num"
            echo "config_hash,$exp_hash"
        } > "$config_dir/hyperparameters.csv"
    fi

    # Function to run one experiment (train, test, extract)
    run_one() {
        local exp_num=$1
        local exp_hash=$2
        local test_version="$3"
        local set_test_version=""
        [ -n "$test_version" ] && set_test_version="--test-version $test_version"
        local pbar_flag=""; [ $no_pbar -eq 1 ] && pbar_flag="--no-pbar"
        local output_dir="$THIS_DIR/train/${exp_hash}/$( [ -n "$test_version" ] && echo "test_${test_version}" || echo "test_0" )"
        mkdir -p "$output_dir"

        echo "[$exp_num/$total_lines] Running $exp_hash"
        {
            $PYTHON "$THIS_DIR/train_model.py" --hash "$exp_hash" $set_test_version $pbar_flag &&
            $PYTHON "$THIS_DIR/test_model.py" --hash "$exp_hash" $set_test_version --epoch best $pbar_flag &&
            $PYTHON "$THIS_DIR/extract_rslt_stats.py" --hash "$exp_hash" $set_test_version --epoch best
        } > "$output_dir/terminal_output.txt" 2>&1

        if [ $? -eq 0 ]; then
            echo "[$(date)] $exp_num SUCCESS" >> "$RESULTS_DIR/sweep_log.txt"
            return 0
        else
            echo "[$(date)] $exp_num FAILED" >> "$RESULTS_DIR/sweep_log.txt"
            return 1
        fi
    }

    # Run experiment (parallel or sequential)
    if [ $MAX_PARALLEL_JOBS -gt 1 ]; then
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do sleep 1; done
        run_one $experiment_num "$exp_hash" "${fields[1]}" &
    else
        run_one $experiment_num "$exp_hash" "${fields[1]}"
        [ $? -ne 0 ] && ((failed++))
    fi

done < /tmp/sweep_combos.txt

# Wait for all background jobs
wait

# Final summary
echo ""
echo "========================================="
echo "Parameter Sweep Complete!"
echo "Total experiments: $experiment_num"
echo "Successful: $((experiment_num - failed))"
echo "Failed: $failed"
echo "========================================="

echo "Sweep finished at $(date)" >> "$RESULTS_DIR/sweep_log.txt"
echo "Total experiments: $experiment_num" >> "$RESULTS_DIR/sweep_log.txt"
echo "Successful: $((experiment_num - failed))" >> "$RESULTS_DIR/sweep_log.txt"
echo "Failed: $failed" >> "$RESULTS_DIR/sweep_log.txt"

rm -f /tmp/sweep_combos.txt
exit $failed