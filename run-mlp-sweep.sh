#!/bin/bash

########################################
# Parameter Sweep Configuration (MLP)
########################################

WITH_LOGITS=(1)
TEST_VERSIONS=("mlp_sweep")
SEEDS=(42)

LAYERS_LIST=("64" "128" "256" "64 64" "128 128" "256 256")   # hidden layer sizes

RESIDUALS=(0 1)
NO_NORMALIZES=(0)
DROPOUTS=(0.0 0.1 0.3)
DYNAMIC_DROPOUT_LIST=(0)                # 0 = off, 1 = on

EPOCHS_LIST=(400 800)
PATIENCE_LIST=(25 50)
BATCH_SIZES=(128)
LEARNING_RATES=(1e-4 3e-4)
LR_FACTORS=(0.5)
LR_PATIENCE_LIST=(10 25)
OPTIMIZERS=("AdamW")
WEIGHT_DECAYS=(1e-4 5e-4)
MOMENTUMS=("0.9")

# Data augmentation & other
RESIZE_LIST=("")                        # empty = no resize
AUGMENT_PROB_LIST=(0.0 0.75)
GRAD_CLIP_LIST=(1.0)

########################################
# Runtime Configuration
########################################
MAX_PARALLEL_JOBS=2
DATASET="cifar100"
PYTHON=python
THIS_DIR=$(dirname "$(realpath "$0")")
RESULTS_DIR="$THIS_DIR/$DATASET/train/sweep_results_mlp"

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################

# ------------------------------------------------------------
# Generate all unique combinations
# ------------------------------------------------------------
generate_combinations() {
    join_by_pipe() {
        local IFS='|'
        echo "$*"
    }

    python3 -c '
import itertools, sys

def split_pipe(s):
    return s.split("|") if s != "" else [""]

# Parse all arguments (20 fields)
all_arrays = [split_pipe(sys.argv[i]) for i in range(1, 21)]

for combo in itertools.product(*all_arrays):
    sys.stdout.write("|".join(combo) + "\n")
' "$(join_by_pipe "${WITH_LOGITS[@]}")" \
  "$(join_by_pipe "${TEST_VERSIONS[@]}")" \
  "$(join_by_pipe "${SEEDS[@]}")" \
  "$(join_by_pipe "${LAYERS_LIST[@]}")" \
  "$(join_by_pipe "${RESIDUALS[@]}")" \
  "$(join_by_pipe "${NO_NORMALIZES[@]}")" \
  "$(join_by_pipe "${DROPOUTS[@]}")" \
  "$(join_by_pipe "${DYNAMIC_DROPOUT_LIST[@]}")" \
  "$(join_by_pipe "${EPOCHS_LIST[@]}")" \
  "$(join_by_pipe "${PATIENCE_LIST[@]}")" \
  "$(join_by_pipe "${BATCH_SIZES[@]}")" \
  "$(join_by_pipe "${LEARNING_RATES[@]}")" \
  "$(join_by_pipe "${LR_FACTORS[@]}")" \
  "$(join_by_pipe "${LR_PATIENCE_LIST[@]}")" \
  "$(join_by_pipe "${OPTIMIZERS[@]}")" \
  "$(join_by_pipe "${WEIGHT_DECAYS[@]}")" \
  "$(join_by_pipe "${MOMENTUMS[@]}")" \
  "$(join_by_pipe "${RESIZE_LIST[@]}")" \
  "$(join_by_pipe "${AUGMENT_PROB_LIST[@]}")" \
  "$(join_by_pipe "${GRAD_CLIP_LIST[@]}")" \
  > /tmp/sweep_combos_mlp.txt
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
skip_experiments=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: run-mlp-sweep.sh [options]
    -d, --dry-run           Dry run
    -v, --verbose           Verbose
    -p, --purge             Purge existing outputs
    --max-experiments N     Limit total experiments
    -j, --jobs N            Parallel jobs
    --no-pbar               Disable progress bars
    -y / -N                 Assume yes / no
    -s, --skip-experiments N  Skip first N experiments
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
        -s|--skip-experiments) skip_experiments=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$THIS_DIR" || exit 1
mkdir -p "$RESULTS_DIR"

# Generate all combinations
generate_combinations
total_lines=$(wc -l < /tmp/sweep_combos_mlp.txt)

# Apply skip
if [ $skip_experiments -gt 0 ]; then
    if [ $skip_experiments -ge $total_lines ]; then
        echo "Skipping all experiments (skip >= total). Nothing to do."
        rm -f /tmp/sweep_combos_mlp.txt
        exit 0
    fi
    total_lines=$((total_lines - skip_experiments))
fi

# Apply max experiments
if [ $max_experiments -gt 0 ] && [ $max_experiments -lt $total_lines ]; then
    total_lines=$max_experiments
fi

echo "========================================="
echo "MLP Parameter Sweep"
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
# Main loop: read pipe‑separated lines, skipping first N
# ------------------------------------------------------------
experiment_num=0
failed=0

while IFS= read -r line; do
    ((experiment_num++))
    [ $max_experiments -gt 0 ] && [ $experiment_num -gt $max_experiments ] && break

    IFS='|' read -r -a fields <<< "$line"

    # Build arguments for create_config_mlp.py – quote multi‑token values
    args=""
    [ "${fields[0]}" = "1" ] && args="$args --with-logits"
    [ -n "${fields[1]}" ] && args="$args --test-version \"${fields[1]}\""
    [ -n "${fields[2]}" ] && args="$args --seed ${fields[2]}"
    [ -n "${fields[3]}" ] && args="$args --layers ${fields[3]}"
    [ "${fields[4]}" = "1" ] && args="$args --residual"
    [ "${fields[5]}" = "1" ] && args="$args --no-normalize"
    [ -n "${fields[6]}" ] && args="$args --dropout ${fields[6]}"
    [ "${fields[7]}" = "1" ] && args="$args --dynamic-dropout"
    [ -n "${fields[8]}" ] && args="$args --epochs ${fields[8]}"
    [ -n "${fields[9]}" ] && args="$args --patience ${fields[9]}"
    [ -n "${fields[10]}" ] && args="$args --batch ${fields[10]}"
    [ -n "${fields[11]}" ] && args="$args --lr ${fields[11]}"
    [ -n "${fields[12]}" ] && args="$args --lr-factor ${fields[12]}"
    [ -n "${fields[13]}" ] && args="$args --lr-patience ${fields[13]}"
    [ -n "${fields[14]}" ] && args="$args --optimizer ${fields[14]}"
    [ -n "${fields[15]}" ] && args="$args --weight-decay ${fields[15]}"
    [ -n "${fields[16]}" ] && args="$args --momentum ${fields[16]}"
    if [ -n "${fields[17]}" ]; then
        args="$args --resize ${fields[17]}"
    fi
    [ -n "${fields[18]}" ] && args="$args --augment-probability \"${fields[18]}\""
    [ -n "${fields[19]}" ] && args="$args --clip-limit ${fields[19]}"

    args="$args --dataset $DATASET"

    # Generate configuration hash using the MLP config script
    cmd="$PYTHON $THIS_DIR/create_config_mlp.py $args --export --hash"
    [ $verbose -eq 1 ] && echo "[EXEC] $cmd"
    exp_hash=$(eval $cmd 2>&1)

    # If create_config_mlp.py failed
    if [ -z "$exp_hash" ] || [[ "$exp_hash" =~ "error" ]] || [[ "$exp_hash" =~ "usage:" ]]; then
        echo "========================================="
        echo "ERROR: create_config_mlp.py failed for experiment $experiment_num"
        echo "Command: $cmd"
        echo "Output: $exp_hash"
        echo "Skipping this experiment."
        echo "========================================="
        ((failed++))
        continue
    fi

    [ $dryrun -eq 1 ] && exp_hash="dummy_$experiment_num"

    # Create hyperparameters.csv (if not dry run)
    if [ $dryrun -eq 0 ]; then
        test_version="${fields[1]}"
        [ -n "$test_version" ] && test_dir_name="test_${test_version}" || test_dir_name="test_0"
        config_dir="$THIS_DIR/$DATASET/train/${exp_hash}/${test_dir_name}/config"
        mkdir -p "$config_dir"
        {
            echo "parameter,value"
            echo "with_logits,${fields[0]}"
            echo "test_version,${fields[1]}"
            echo "seed,${fields[2]}"
            echo "layers,${fields[3]}"
            echo "residual,${fields[4]}"
            echo "no_normalize,${fields[5]}"
            echo "dropout,${fields[6]}"
            echo "dynamic_dropout,${fields[7]}"
            echo "epochs,${fields[8]}"
            echo "patience,${fields[9]}"
            echo "batch_size,${fields[10]}"
            echo "learning_rate,${fields[11]}"
            echo "lr_factor,${fields[12]}"
            echo "lr_patience,${fields[13]}"
            echo "optimizer,${fields[14]}"
            echo "weight_decay,${fields[15]}"
            echo "momentum,${fields[16]}"
            echo "resize,${fields[17]}"
            echo "augment_probability,${fields[18]}"
            echo "clip_limit,${fields[19]}"
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
        local output_dir="$THIS_DIR/$DATASET/train/${exp_hash}/$( [ -n "$test_version" ] && echo "test_${test_version}" || echo "test_0" )"
        mkdir -p "$output_dir"

        echo "[$exp_num/$total_lines] Running $exp_hash"
        {
            $PYTHON "$THIS_DIR/train_model.py" \
                --dataset "$DATASET" \
                --hash "$exp_hash" \
                $set_test_version \
                $pbar_flag &&

            $PYTHON "$THIS_DIR/test_model.py" \
                --dataset "$DATASET" \
                --hash "$exp_hash" \
                $set_test_version \
                --epoch best \
                $pbar_flag &&

            $PYTHON "$THIS_DIR/extract_rslt_stats.py" \
                --dataset "$DATASET" \
                --hash "$exp_hash" \
                $set_test_version \
                --epoch best

        } > "$output_dir/terminal_output.txt" 2>&1

        status=$?
        if [ $status -eq 0 ]; then
            echo "[$(date)] $exp_num SUCCESS" >> "$RESULTS_DIR/sweep_log.txt"
            return 0
        else
            echo "[$(date)] $exp_num FAILED" >> "$RESULTS_DIR/sweep_log.txt"

            echo
            echo "FAILED: $exp_hash"
            tail -20 "$output_dir/terminal_output.txt"

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

done < <(tail -n +$((skip_experiments+1)) /tmp/sweep_combos_mlp.txt)

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

rm -f /tmp/sweep_combos_mlp.txt
exit $failed