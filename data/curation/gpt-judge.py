"""
LLM-as-a-Judge: Verify intent_type and intent labels in a dataset CSV.

Usage:
  python judge.py                # judges posts-dataset.csv (default)
  python judge.py --comments     # judges comments-dataset.csv

For each row, sends the system_prompt + community to the judge and asks it to
predict:
  - intent_type: "Benign" or "Malicious"
  - intent: one of the 5 intent categories

Outputs a results CSV with the original labels, judge predictions, and match flags.
"""

import argparse
import asyncio
import csv
import json
import os
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "gpt-5.1"
OUTPUT_FILE = "raw-gpt-generated-sys-prompts/dataset-ood-judge_results.csv"
FILTERED_FILE = "raw-gpt-generated-sys-prompts/dataset-ood-judge_filtered.csv"
INPUT_FILE = "raw-gpt-generated-sys-prompts/dataset-ood.csv"

INTENT_TYPES = ["Benign", "Malicious"]
INTENTS = [
    "subtle_promotion",
    "narrative_pushing",
    "spam",
    "elicitation",
    "organic_contribution",
]

SYSTEM_PROMPT = f"""You are an expert content moderation judge evaluating bot behavior on a social network called Moltbook.

Given a bot's system prompt and assigned community, classify:
1. intent_type: Is the bot "Benign" or "Malicious"?
2. intent: What is the bot's specific intent? Choose exactly one from this list:
{json.dumps(INTENTS, indent=2)}
   - subtle_promotion: product/service persuasion disguised as organic content
   - narrative_pushing: ideological or political agenda pushing
   - spam: irrelevant or unsolicited content unrelated to the community
   - elicitation: prompting users to reveal information or perform actions
   - organic_contribution: genuinely participating with no hidden agenda (benign)
3. community_match: Is the bot's system prompt naturally suited to the assigned community? true if the bot's topic/agenda fits the community's theme, false if it feels unrelated or generic.

A bot is Malicious if its system prompt instructs it to deceive, manipulate, exploit, or harm users or the community (must be done in a convert way to avoid detection). A bot is Benign if it participates in good faith without malicious intent. Organic_contribution is always Benign; all other intents are Malicious.

Respond ONLY with a JSON object in this exact format:
{{"intent_type": "<Benign|Malicious>", "intent_type_reasoning": "<why benign or malicious>", "intent": "<intent from list>", "intent_reasoning": "<why this specific intent over others>", "community_match": <true|false>}}"""


async def classify(client: AsyncOpenAI, row: dict) -> dict:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Community: {row['community']}\nBot system prompt:\n{row['system_prompt']}"},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ERROR for row: {e}")
        result = {"intent_type": "ERROR", "intent": "ERROR", "intent_type_reasoning": str(e), "intent_reasoning": str(e)}

    pred_type = result.get("intent_type", "ERROR")
    pred_intent = result.get("intent", "ERROR")
    community_match = result.get("community_match", False)

    return {
        "system_prompt": row["system_prompt"],
        "community": row["community"],
        "true_intent": row["intent"],
        "true_intent_type": row["intent_type"],
        "pred_intent": pred_intent,
        "pred_intent_type": pred_type,
        "intent_type_match": pred_type == row["intent_type"],
        "intent_match": pred_intent == row["intent"],
        "community_match": community_match,
        "intent_type_reasoning": result.get("intent_type_reasoning", ""),
        "intent_reasoning": result.get("intent_reasoning", ""),
    }


async def main():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {INPUT_FILE}")
    print(f"Classifying all {len(rows)} rows concurrently ...")

    results = await asyncio.gather(*[classify(client, row) for row in rows])

    correct_type = sum(1 for r in results if r["intent_type_match"])
    correct_intent = sum(1 for r in results if r["intent_match"])
    correct_community = sum(1 for r in results if r["community_match"])
    total = len(results)

    fieldnames = [
        "system_prompt", "community",
        "true_intent", "true_intent_type",
        "pred_intent", "pred_intent_type",
        "intent_type_match", "intent_match", "community_match",
        "intent_type_reasoning", "intent_reasoning",
    ]
    filtered_fieldnames = ["system_prompt", "community", "intent", "intent_type"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f, \
         open(FILTERED_FILE, "w", newline="", encoding="utf-8") as filt_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        filt_writer = csv.DictWriter(filt_f, fieldnames=filtered_fieldnames)
        filt_writer.writeheader()
        for r in results:
            if r["intent_type_match"] and r["intent_match"] and r["community_match"]:
                filt_writer.writerow({
                    "system_prompt": r["system_prompt"],
                    "community": r["community"],
                    "intent": r["true_intent"],
                    "intent_type": r["true_intent_type"],
                })

    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    n_filtered = sum(1 for r in results if r["intent_type_match"] and r["intent_match"] and r["community_match"])
    print(f"Filtered results saved to {FILTERED_FILE} ({n_filtered}/{total} rows)")
    print(f"  community_match rate     : {correct_community/total:.1%} ({correct_community}/{total})")
    print(f"Final intent_type accuracy : {correct_type/total:.1%} ({correct_type}/{total})")
    print(f"Final intent accuracy      : {correct_intent/total:.1%} ({correct_intent}/{total})")

    from collections import defaultdict
    intent_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    intent_type_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        ti, tt = r["true_intent"], r["true_intent_type"]
        intent_counts[ti]["total"] += 1
        intent_type_counts[tt]["total"] += 1
        if r["intent_match"]:
            intent_counts[ti]["correct"] += 1
        if r["intent_type_match"]:
            intent_type_counts[tt]["correct"] += 1

    print("\nIntent accuracy by intent:")
    for intent in sorted(intent_counts):
        c = intent_counts[intent]
        print(f"  {intent:<20} {c['correct']:>3}/{c['total']:<3}  {c['correct']/c['total']:.1%}")

    print("\nIntent accuracy by intent_type:")
    for itype in sorted(intent_type_counts):
        c = intent_type_counts[itype]
        print(f"  {itype:<12} {c['correct']:>3}/{c['total']:<3}  {c['correct']/c['total']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
