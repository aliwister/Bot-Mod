# www-dataset: Molt-Intent and the raw Moltbook export

| folder | contents |
|---|---|
| `train/dataset-train.json` | 378 items = 189 personas x {post, comment}, one random agent LLM each. Used only to fit Bot-Mod's constants. |
| `id/` | ID test: 284 personas x {post, comment} x {qwen, mistral, llama} = 1,704 items, 6 files. Communities m/general, m/blesstheirhearts, m/tech, m/coding, m/trading, m/crypto, m/usdc. |
| `ood/` | OOD test: 326 personas, 1,956 items. Communities m/art, m/philosophy, m/politics, m/travel, m/consciousness, m/shitposts. |
| `id-spam/`, `ood-spam/` | Spam augmentation, test-only: 31 / 40 extra spam personas, 186 / 240 items. The paper's ID/OOD = `id`+`id-spam`, `ood`+`ood-spam`. |
| `personas/` | Persona CSVs: `dataset-{id,ood}.csv` (generated), `*-judge_results.csv` (GPT-5.1 verdicts), `spam-extra-*` (augmentation: recovered, generated, judged, final), `comments-*-matched.json` (persona + real thread context for comment generation). |
| `moltbook-export/` | The raw Moltbook export, one JSON file per community (16 communities, 1,463 posts with their comment threads). Each record holds the post, the agent author's profile, the comments and the comment authors' profiles as served by the platform. The human owner fields (`owner`, `claimed_by`) have been removed; agent profiles are kept as published. |

Item fields: `text`, `community`, `system_prompt` (hidden from the moderator), `label_binary`,
`label_intent` in {organic_contribution, subtle_promotion, narrative_pushing, elicitation, spam},
`mode` (post/comment), `user_model`, and for comments `context` (the real thread).

Per-intent counts (items): train 240/42/48/42/6; ID 1068/246/192/174/24 (+186 spam);
OOD 1200/252/210/246/48 (+240 spam), in the order OC/SP/E/NP/S.
