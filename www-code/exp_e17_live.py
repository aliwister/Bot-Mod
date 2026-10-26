"""Bot-Mod and Bot-Mod-judge against a live agent session on a test split (the paper's protocol).

Each item is ONE conversation with the agent (interrogation.ask_agent): the three probes, one of
four wordings per technique chosen from the item id (exp_e16_multiturn.varied_question), in the
fixed technique order. The moderator side is stateless:

  bot-mod            each answer scored alone, log M(a_k | persona(h), content, q_k) for every h
                     (exp_e3_teacher_forced_likelihood.loglik); constants from fit_varied_bank.py
  bot-mod-judge      a judge reads content + the three question-answer pairs (interrogation.read_intent)

Both arms read the SAME transcript, so they differ only in how the answers are read. The content
prior is reused from the matrix stage (e12_matrix_<split>_own.json), so every arm starts from the
same prior.

    BOTMOD_PORT_BASE=18000 python exp_e17_live.py run <split>        split in id, id_spam, ood, ood_spam
        -> www-results/e17_live_<split>_varied.json           the transcripts
           www-results/e12_matrix_<split>_varied_live.json    Bot-Mod's per-answer likelihoods
           www-results/e17_fbj_varied_live_<split>.json       Bot-Mod-judge's verdicts
    python exp_e12_head_to_head.py score id+spam    (and ood+spam)
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import enrich_content, _is_organic  # noqa: E402
from exp_e3_teacher_forced_likelihood import loglik, tok  # noqa: E402
from exp_e16_multiturn import varied_question, TURNS  # noqa: E402
from interrogation import ask_agent, read_intent, _rec  # noqa: E402
from exp_e12_head_to_head import LOGS, load_items  # noqa: E402

WORKERS = 48
SUF = "_varied"


def converse(row):
    content, mode = enrich_content(row)
    community = row["community"].strip()
    turns, probes = [], {}
    for k in range(TURNS):
        p, qtext = f"s{k + 1}", varied_question(row["_item_id"], k)
        a = ask_agent(row, content, turns, qtext)             # live session: all earlier turns in context
        if not a:
            return None
        lls, n = [], 0
        for h in INTENTS:                                      # stateless moderator: this answer alone
            ll, n = loglik(content, community, qtext, a, h)
            if ll is None:
                return None
            lls.append(ll)
        turns.append((qtext, a))
        probes[p] = {"loglik": lls, "answer_tokens": n, "answer_chars": len(a)}
    t, tm = read_intent(content, community, turns)
    return {"item_id": row["_item_id"], "mode": mode, "community": community,
            "src": row["_src"], "user_model": row.get("user_model", ""),
            "y": INTENTS.index(row["label_intent"].strip().lower()),
            "y_binary": row["label_binary"].strip().upper(),
            "intent_type": row["label_binary"].strip().upper(),
            "intent": row["label_intent"].strip().lower(),
            "turns": [{"q": q, "answer": a} for q, a in turns], "probes": probes,
            "fbj": {"t_judge": t, "t_malicious": tm}}


def stage_run(split):
    tok()
    items = load_items(split)
    if os.environ.get("BOTMOD_E17_LIMIT"):                    # smoke test
        items = items[:int(os.environ["BOTMOD_E17_LIMIT"])]
    prior = {r["item_id"]: r["prior_raw"]
             for r in json.load(open(os.path.join(LOGS, f"e12_matrix_{split}_own.json")))["records"]}
    t0, done = time.time(), [0]
    print(f"[{split}] live interrogation: {len(items)} items x {TURNS} turns", flush=True)

    def one(row):
        try:
            rec = converse(row)
        except Exception as e:                                 # noqa: BLE001
            print(f"  ! {row['_item_id']}: {type(e).__name__}: {e}", flush=True)
            rec = None
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs = [r for r in ex.map(one, items) if r is not None]
    cfg = {"split": split, "agent": "live session (interrogation.ask_agent)", "probes": [f"s{k + 1}" for k in range(TURNS)],
           "questions": "varied", "n_requested": len(items), "n": len(recs), "seconds": time.time() - t0}
    json.dump({"config": cfg, "records": recs}, open(os.path.join(LOGS, f"e17_live_{split}{SUF}.json"), "w"))
    # Bot-Mod: matrix records in the replay format, prior reused from the content-only matrix
    mat = [{"item_id": r["item_id"], "y": r["y"], "y_binary": r["y_binary"], "mode": r["mode"],
            "community": r["community"], "src": r["src"], "user_model": r["user_model"],
            "prior_raw": prior[r["item_id"]], "probes": r["probes"]}
           for r in recs if r["item_id"] in prior]
    json.dump({"config": {**cfg, "n_full": len(mat)}, "records": mat},
              open(os.path.join(LOGS, f"e12_matrix_{split}{SUF}_live.json"), "w"))
    # Bot-Mod-judge: per-item verdict rows
    fbj = [_rec(r, "BENIGN" if _is_organic(r["fbj"]["t_judge"]) else "MALICIOUS",
                r["fbj"]["t_judge"] if not _is_organic(r["fbj"]["t_judge"]) else r["fbj"]["t_malicious"])
           for r in recs]
    json.dump({"config": cfg, "arms": {"fixed-bank-judge": fbj}},
              open(os.path.join(LOGS, f"e17_fbj{SUF}_live_{split}.json"), "w"))
    print(f"[{split}] {len(recs)}/{len(items)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        stage_run(sys.argv[2])
    else:
        sys.exit("want: run <split>")
