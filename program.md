# autoresearch

This is an experiment to have the LLM do its own research.

## Runtime

The python uv environment is: ../.venv

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `eval.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/cache` contains data files. 
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes**  (wall clock training time, excluding startup/compilation). You launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: prompt text, prompt structure, hyperparameters, training loop, prompting approach, sampling approach, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, general logic, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_f1` function in `prepare.py` is the ground truth metric.
- Modify the test harness in `eval.py`.

**The goal is simple: get the lowest val_f1.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, moderator functions, the hyperparameters, prompt strategy (can use SOTA methods from literature), prompt text, prompt strategy, convergence strategy, number of iterations, etc. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_f1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_f1 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_f1 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is. After the baseline training completes, **immediately run the test evaluation** (`python3 eval.py > eval_run.log 2>&1`) to establish the baseline test metrics. This baseline is critical for all future acceptance decisions.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_f1:           0.9979
f1_binary:        0.9588
f1_categorical:   0.9121
val_f1_zs:        0.7878
f1_zs:            0.8766
f1_cat_zs:        0.6773
total_seconds:    325.9
```


## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_f1	  f1_bin  f1_cat  val_f1_zs	  f1_zs_bin    f1_zs_cat  memory_gb	status
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `eval`, `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_f1	  f1_bin   f1_cat   val_f1_zs	  f1_zs_bin  f1_zs_cat  memory_gb	status	description
a1b2c3d	0.997900	  0.8738   0.8738   0.8738       0.8738     0.8738    44.0	      eval	baseline
b2c3d4e	0.993200	  0.8638   0.8638   0.8638       0.8638     0.8638    44.2	      eval	change mod system prompt
c3d4e5f	1.005000	  0.7738   0.7738   0.7738       0.7738     0.7738    44.0	      discard	change probe propmt
d4e5f6g	0.000000	  0.5738   0.5738   0.5738       0.5738     0.5738    0.0	      crash	double probe iterations
```

### Test Results Logging

When test evaluation is run (after an improvement on training), also log to `test-results.tsv` (tab-separated).

The test TSV has a header row and these columns:

Example:
```
commit	val_f1	  f1_bin   f1_cat   val_f1_zs	  f1_zs_bin  f1_zs_cat  memory_gb	status	description
a1b2c3d	0.997900	  0.8738   0.8738   0.8738       0.8738     0.8738    44.0	      keep	baseline
b2c3d4e	0.993200	  0.8638   0.8638   0.8638       0.8638     0.8638    44.2	      discard	change mod system prompt
d4e5f6g	0.000000	  0.5738   0.5738   0.5738       0.5738     0.5738    0.0	      crash	double probe iterations
```


**NOTE**: Do not commit `results.tsv` or `test-results.tsv` — leave them untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_f1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the training results in `results.tsv` as soon as the values are available (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. **Test Evaluation**: Only if the training result improves over the current best, mark it as `eval` then run `python3 eval.py > eval_run.log 2>&1` to evaluate on the test set. Do NOT run eval.py if training did not improve, mark it as `discard` in results.tsv.
9. **Log Test Results**: Record the test metrics in `test-results.tsv` with all results and the commit hash.
10. **Acceptance Criteria**: Only accept the commit if it also wins on eval.py (i.e., test metrics must also improve over the current best). If either metric decreases on the test set, mark as `discard` in test-results.tsv and revert.
11. If both test metrics improved over the current best, mark as `keep` in test-results.tsv and "advance" the branch, keeping the git commit.
12. If training did not improve, OR if eval.py metrics did not improve (even if training improved), git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~30 minutes then you can run approx 2/hour, for a total of about 78 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!