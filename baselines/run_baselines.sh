#!/bin/bash
# Run each baseline 3 times on test + ood, in parallel per baseline (all 3 runs for a
# baseline run simultaneously). Baselines run sequentially so the backend isn't overwhelmed.
set +e
cd /run/user/1000/autoresearch3/baselines

BASELINES=(zero_shot self_consistency cot self_refine)

for bl in "${BASELINES[@]}"; do
    dir="../logs/baselines/${bl}"
    mkdir -p "$dir"

    echo "================================================" | tee -a ../baselines_runner.log
    echo "[$(date -Iseconds)] baseline=${bl}" | tee -a ../baselines_runner.log

    pids=()
    for run in 1 2 3; do
        out="$dir/run${run}.log"
        if [[ -s "$out" ]] && grep -q "^time:" "$out" && [[ $(grep -c "^time:" "$out") -ge 12 ]]; then
            echo "  run$run already complete, skipping" | tee -a ../baselines_runner.log
            continue
        fi
        echo "  [$(date -Iseconds)] start run $run -> $out" | tee -a ../baselines_runner.log
        python3 "${bl}.py" both > "$out" 2>&1 &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
        echo "  [$(date -Iseconds)] pid $pid exit=$?" | tee -a ../baselines_runner.log
    done
done

echo "[$(date -Iseconds)] ALL BASELINES DONE" | tee -a ../baselines_runner.log
