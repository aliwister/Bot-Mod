import json
import csv, asyncio, copy, random
random.seed(42)
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI, OpenAI

_LLM_ALIASES = {
    "qwen":    "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
}

def resolve_model(name: str) -> str:
    return _LLM_ALIASES.get(name.lower(), name)

_BASE_URL = {}
_BASE_URL['Qwen/Qwen3-8B'] = "http://localhost:8000/v1"
_BASE_URL["mistralai/Mistral-7B-Instruct-v0.3"] = "http://localhost:8001/v1"
_BASE_URL["meta-llama/Llama-3.1-8B-Instruct"] = "http://localhost:8002/v1"

_clients: dict[str, OpenAI] = {}

def _base_url(model: str) -> str:
    model = resolve_model(model)
    if model in _BASE_URL:
        return _BASE_URL[model]


def _client(model: str) -> OpenAI:
    model = resolve_model(model)
    url = _base_url(model)
    if url not in _clients:
        _clients[url] = OpenAI(base_url=url, api_key="")
    return _clients[url]


# ── SYNC (single call) ────────────────────────────────────────────────────────

def call_llm_messages(messages: list, temperature: float, model: str,
                      max_new_tokens: int = 1024, json_mode: bool = False) -> str:
    model = resolve_model(model)
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_new_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _client(model).chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def call_llm(system_prompt: str, user_message: str, temperature: float,
             model: str, max_new_tokens: int = 1024, json_mode: bool = False) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    return call_llm_messages(messages, temperature, model, max_new_tokens, json_mode=json_mode)


# ── ASYNC BATCH ───────────────────────────────────────────────────────────────

async def call_llm_async(messages: list, temperature: float, model: str,
                         max_new_tokens: int = 1024) -> str:
    model = resolve_model(model)
    async with AsyncOpenAI(base_url=_base_url(model), api_key="") as client:
        response = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content.strip()


def call_llm_batch(messages_list, temperature, model: str, max_new_tokens: int = 1024) -> list[str]:
    """
    Send a batch of message lists concurrently via asyncio.gather.
    Returns responses in the same order as `messages_list`.
    """
    async def _gather():
        return await asyncio.gather(*[
            call_llm_async(msgs, temperature, model, max_new_tokens)
            for msgs in messages_list
        ])
    return asyncio.run(_gather())


INTENTS = [
    "subtle_promotion", #    (product/service persuasion) ~ ex found in m/cryoto m/usdc for external wallet, website, exchange, etc.
    "narrative_pushing", #   (ideological/political agenda) ~ actually found in m/tech or relevant like why these agent setup too use, blah blah
    "spam", #                (irrelevant or spam content) ~ broad category for unrelated items to the submolt
    "elicitation", #     (prompting users to reveal info) ~ ex: asking to revel command/port information/etc.
    "orangic_contriubtion" # (genuinely participating)
]

INTENT_TYPES = ["Benign", "Malicious"]
_NOTHINK_MODELS = {"Qwen/Qwen3-8B"}


MOD_MODEL="Qwen/Qwen3-8B"
def llm_mod(system_prompt, user_prompt, temp=0.7):
    suffix = " /nothink" if MOD_MODEL in _NOTHINK_MODELS else ""
    return call_llm(system_prompt, user_prompt + suffix, temp, model=MOD_MODEL)



class UserBot:
    def __init__(self, sub, user_system_prompt, mode: str = "post", context: str = "", model=None, pregenerated_text: str = None):
        self.model = model or random.choice(list(_BASE_URL.keys()))
        self.user_system_prompt = user_system_prompt
        self.mode = mode

        self._user_messages: list[dict] = [
            {"role": "system", "content": user_system_prompt}
        ]

        # If pre-generated text is provided, use it directly
        if pregenerated_text is None:
            raise Exception("Pre-generated text must be provided for UserBot.")
        self.M = pregenerated_text
        # Add the initial prompt and response to message history for consistency
        if mode == "comment":
            prompt = (
                f"Post:\n{context}\n\n"
                f"Comment on the post"
            )
        else:
            action = "Post to"
            prompt = (
                f"{action} {sub}"
            )
        self._user_messages.append({"role": "user", "content": prompt})
        self._user_messages.append({"role": "assistant", "content": pregenerated_text})



    def _call_user(self, prompt: str, temp: float = 0.3) -> str:
        suffix = " /nothink" if self.model in _NOTHINK_MODELS else ""
        self._user_messages.append({"role": "user", "content": prompt+suffix})
        response = call_llm_messages(self._user_messages, temp, model=self.model)
        self._user_messages.append({"role": "assistant", "content": response})
        return response

    def llm_sampler(self, prompt):
        response = self._call_user(prompt)
        return response.strip().lower()

    def user_message(self, prompt):
        msg = self._call_user(prompt)
        print(msg)
        return msg.strip()

# --- CSV Evaluation ---
TRAIN_DATA = "cache/train-generated.json"  # Pre-generated merged data
TEST_DATA = "cache/test-generated.json"  # Pre-generated merged data

def metric(results):
    # Binary F1: malicious vs benign
    tp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "MALICIOUS")
    fp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "BENIGN")
    fn = sum(1 for r in results if r["verdict"] == "BENIGN"    and r["intent_type"] == "MALICIOUS")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1_binary = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Macro categorical F1: predicted intent (t) vs true intent
    mal = [r for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "MALICIOUS"]
    classes = {r["intent"] for r in mal}
    f1_per_class = []
    for cls in classes:
        tp_c = sum(1 for r in mal if r["t"] == cls and r["intent"] == cls)
        fp_c = sum(1 for r in mal if r["t"] == cls and r["intent"] != cls)
        fn_c = sum(1 for r in mal if r["t"] != cls and r["intent"] == cls)
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        f1_per_class.append(2 * p * r / (p + r) if (p + r) else 0.0)
    f1_categorical = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0

    return f1_binary, f1_categorical

def _save_results_csv(results, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_intent", "pred_intent", "true_intent_type", "pred_intent_type", "submolt", "M", "context"])
        for r in results:
            writer.writerow([r["intent"], r["t"], r["intent_type"], r["verdict"], r["community"], r["M"], r["context"]])

def evaluate_f1_train(mod, user_model=None, filename=None):
    return evaluate_f1(mod, user_model=user_model, filename=TRAIN_DATA)

def evaluate_f1_test(mod, user_model=None, filename=None):
    return evaluate_f1(mod, user_model=user_model, filename=TEST_DATA)

def evaluate_f1(mod, user_model=None, filename=None):
    # Load pre-generated merged data
    with open(filename, encoding="utf-8") as f:
        rows = json.load(f)

    # Ensure context is properly formatted for comments
    for row in rows:
        if row["mode"] == "comment" and "context" in row:
            # Keep context as dict for now, will serialize when needed
            if isinstance(row["context"], str):
                row["context"] = json.loads(row["context"])
        elif row["mode"] == "post":
            row["context"] = ""

    def process_row(args):
        idx, row = args
        community     = row["community"].strip()
        system_prompt = row["system_prompt"].strip()
        truth         = row["label_binary"].strip().lower()  # Changed from intent_type
        intent        = row["label_intent"].strip().lower()  # Changed from intent
        mode          = row.get("mode", "post").strip() or "post"
        context       = row.get("context", "")
        generated_text = row.get("text", "").strip()  # Pre-generated text
        llm_used      = row.get("user_model", "unknown")  # LLM that generated this

        print(f"[{idx+1}/{len(rows)}]  community={community}  mode={mode}  intent={intent}  truth={truth}  llm={llm_used}")

        m = copy.deepcopy(mod)
        # Serialize context for comment mode
        if mode == "comment" and isinstance(context, dict):
            context_str = json.dumps(context, ensure_ascii=False)
        else:
            context_str = context if isinstance(context, str) else ""

        # Create UserBot with pre-generated text from train_generated_merged.json
        user = UserBot(
            community,
            system_prompt,
            mode=mode,
            context=context_str,
            model=user_model or llm_used,
            pregenerated_text=generated_text  # Use pre-generated text, skip generation
        )
        m._init_hypothesis(user.M, community)

        for i in range(m.max_iterations):
            critique = m._refine_hypothesis()   # samples t, y internally
            question = m._generate_probe(critique)
            response = user._call_user(question)
            m.P.append({"Critique": critique, "Q": question, "R": response})
            if m.is_converged():
                break
        m._refine_hypothesis()  # incorporate the last response
        verdict = m.y
        correct = verdict == truth
        print(f"[{idx+1}]  verdict={verdict}  truth={truth}  correct={correct}")
        return (
            {"verdict": verdict.upper(), "intent_type": truth.upper(), "correct": correct, "intent": intent.lower(), "t": m.t.strip().lower(), "community": community, "M": user.M, "context": context_str},
            {"verdict": m.y0.upper(), "intent_type": truth.upper(), "correct": m.y0 == truth, "intent": intent.lower(), "t": getattr(m, "t0", m.t).strip().lower(), "community": community, "M": user.M, "context": context_str},
        )

    with ThreadPoolExecutor() as executor:
        pairs = list(executor.map(process_row, enumerate(rows)))

    results          = [p[0] for p in pairs]
    results_zeroshot = [p[1] for p in pairs]

    #if filename:
    #    _save_results_csv(results, filename)
    #    _save_results_csv(results_zeroshot, filename.replace(".csv", "_zeroshot.csv"))

    f1, f1_cat    = metric(results)
    f1_zs, f1_cat_zs = metric(results_zeroshot)
    _lambda = 0.7
    f1_merged = _lambda * f1 + (1 - _lambda) * f1_cat
    f1_zs_merged = _lambda * f1_zs + (1 - _lambda) * f1_cat_zs
    print(f"F1-score (iterative): {f1:.4f} (zero-shot): {f1_zs:.4f}")
    return f1_merged, {'f1_bin': f1, 'f1_cat': f1_cat, 'f1_zs_merged': f1_zs_merged, 'f1_zs': f1_zs, 'f1_cat_zs': f1_cat_zs}

if __name__ == "__main__":
    print("Done! Ready to train.")