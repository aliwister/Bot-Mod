"""
Run moderator-user traces and render each directly to a PDF figure.

No text-file intermediate: trace data is converted to a Transcript in memory,
then rendered with matplotlib.

Usage:
    python generate_conversation_pdfs.py [--start N] [--count N] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

# Prefer Liberation Sans (metric-compatible with Arial, visually lighter than
# DejaVu). Falls back cleanly to DejaVu if Liberation isn't installed.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "Helvetica", "DejaVu Sans",
]

# prepare/train live at the repo root; make them importable when this
# script is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import Counter

from prepare import UserBot, llm_mod
from train import INTENTS, ModeratorBot


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------
#
# This script does *not* re-implement any of the moderator's prompting or
# sampling. It runs the real `ModeratorBot` from train.py and only observes
# the LLM calls it makes via a lightweight wrapper around `llm_mod`. The
# trace rendered to PDF is reconstructed entirely from (a) the captured
# request/response log and (b) the state the moderator exposes on `mod`
# (`mod.y`, `mod.t`, `mod.P`).


def _fmt_votes(votes: list[str]) -> str:
    counts = Counter(votes).most_common()
    return ", ".join(f"{v} ×{n}" for v, n in counts)


def _match_intent(response: str) -> str:
    """Mirror the (response -> intent) mapping used inside ModeratorBot._vote_intent."""
    v = response.strip().lower()
    for intent in INTENTS:
        if intent in v:
            return intent
    return v


class _LoggingLLM:
    """Callable wrapper around `llm_mod` that records every call the moderator
    makes. The moderator uses `self.llm_mod(...)`, so by passing an instance of
    this wrapper as the `llm_mod` argument to `ModeratorBot(...)` we can recover
    every (system_prompt, user_prompt, response, temp) tuple without touching
    train.py."""

    def __init__(self, llm_mod_fn):
        self._llm_mod = llm_mod_fn
        self.calls: list[dict] = []

    def __call__(self, system_prompt, user_prompt, temp=0.7, **kwargs):
        response = self._llm_mod(system_prompt, user_prompt, temp=temp, **kwargs)
        # prepare.llm_mod hard-codes the moderator model; mirror it here so
        # the captured log is self-describing.
        self.calls.append({
            "model":    "Qwen/Qwen3-8B",
            "system":   system_prompt,
            "user":     user_prompt,
            "temp":     temp,
            "response": response,
        })
        return response

    def votes_since(self, start: int) -> list[str]:
        return [_match_intent(c["response"]) for c in self.calls[start:]]


def run_trace(data):
    community     = data["community"]
    system_prompt = data["system_prompt"]
    generated_text = data["text"]
    mode   = data["mode"]
    truth  = data["label_binary"]
    intent = data["label_intent"]

    try:
        post_data = json.loads(data["text"])
        title = post_data.get("title", "No Title")
        content = post_data.get("content", generated_text)
    except Exception:
        title = "Post"
        content = generated_text

    print(f"\nProcessing: {title[:50]}...")

    # Moderator is Qwen3-8B (pinned by prepare.llm_mod). User backend is
    # whichever LLM generated this row's text, as recorded in data["user_model"].
    user = UserBot(community, system_prompt, mode=mode, context="",
                   model=data["user_model"], pregenerated_text=generated_text)
    log = _LoggingLLM(llm_mod)
    mod = ModeratorBot(log)

    # -- init: all LLM calls made during _init_hypothesis are intent votes --
    n = len(log.calls)
    mod._init_hypothesis(user.M, community)
    initial_votes = log.votes_since(n)

    trace = []
    trace.append({"type": "user", "title": title, "content": content,
                  "is_initial": True})
    trace.append({
        "type": "moderator_thinking",
        "text": (f"Initial assessment (self-consistency, n={len(initial_votes)})\n"
                 f"votes: {_fmt_votes(initial_votes)}\n"
                 f"→ {mod.y} ({mod.t})"),
    })

    # -- probing loop: ModeratorBot drives refinement + probe generation --
    for _ in range(mod.max_iterations):
        n = len(log.calls)
        critique = mod._refine_hypothesis()
        votes = log.votes_since(n)
        trace.append({
            "type": "moderator_thinking",
            "text": (f"Hypothesis refinement (self-consistency, n={len(votes)})\n"
                     f"votes: {_fmt_votes(votes)}\n"
                     f"→ Hypothesis: {mod.y} → {mod.t}"),
        })

        trace.append({"type": "prompt_label", "text": "PROBE PROMPT · follow-up"
                                                       if mod.P
                                                       else "PROBE PROMPT · opening"})
        question = mod._generate_probe(critique)
        trace.append({"type": "moderator", "text": question})

        response = user._call_user(question)
        trace.append({"type": "user", "text": response})

        mod.P.append({"Critique": critique, "Q": question, "R": response})
        if mod.is_converged():
            break

    n = len(log.calls)
    mod._refine_hypothesis()
    votes = log.votes_since(n)
    trace.append({
        "type": "moderator_thinking",
        "text": (f"Hypothesis refinement (self-consistency, n={len(votes)})\n"
                 f"votes: {_fmt_votes(votes)}\n"
                 f"→ Hypothesis: {mod.y} → {mod.t}"),
    })

    if hasattr(mod, "finalize_intent"):
        n = len(log.calls)
        mod.finalize_intent()
        final_votes = log.votes_since(n)
        trace.append({
            "type": "moderator_thinking",
            "text": (f"Final intent vote (self-consistency, n={len(final_votes)})\n"
                     f"votes: {_fmt_votes(final_votes)}\n"
                     f"→ {mod.y} ({mod.t})"),
        })

    trace.append({
        "type": "verdict",
        "predicted":   mod.y.upper(),
        "intent":      mod.t,
        "truth":       truth.upper(),
        "true_intent": intent,
        "correct":     mod.y == truth,
    })

    return trace, community, title


# ---------------------------------------------------------------------------
# Transcript model
# ---------------------------------------------------------------------------

EMOJI_RX = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF‍️]"
)


@dataclass
class Turn:
    kind: Literal["initial", "internal", "mod_q", "user_r"]
    body: str
    title: str = ""


@dataclass
class Transcript:
    community: str = ""
    turns: list[Turn] = field(default_factory=list)
    verdict: dict = field(default_factory=dict)


def _strip_emoji(s: str) -> str:
    return EMOJI_RX.sub("", s)


def _clean_content(content: str) -> str:
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content)
    content = _strip_emoji(content)
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


def _extract_initial(raw_text: str, title_hint: str, content_hint: str) -> tuple[str, str]:
    """Recover (title, content) from a post payload.

    run_trace's upstream `json.loads(data["text"])` sometimes fails on
    malformed JSON (nested quotes etc.) and falls back to title="Post" and
    content=raw_text. In that case we retry here: find a `{...}` block,
    parse it, and fall back to regex extraction of "title"/"content"
    fields if json.loads still fails.
    """
    if title_hint and title_hint != "Post":
        return title_hint, content_hint

    body = content_hint or raw_text or ""
    title, content = title_hint, body

    m = re.search(r"\{.*\}", body, re.DOTALL)
    if m:
        raw = m.group(0)
        try:
            data = json.loads(raw)
            title = data.get("title", title)
            content = data.get("content", body)
        except Exception:
            mt = re.search(r'"title"\s*:\s*"([^"]*)"', raw)
            if mt:
                title = mt.group(1)
            mc = re.search(r'"content"\s*:\s*"(.*)"\s*\}\s*$', raw, re.DOTALL)
            if mc:
                content = mc.group(1)
    return title, content


def _clean_internal(text: str) -> str:
    """Tidy up moderator_thinking strings for display.

    Replace single newlines with paragraph breaks so wrap_to_width doesn't
    collapse multi-line content (vote lines + arrow lines) onto one visual line.
    """
    return text.strip().replace("\n", "\n\n")


def trace_to_transcript(trace, community: str) -> Transcript:
    t = Transcript(community=community)
    pending_prompt_label: str | None = None

    for item in trace:
        kind = item["type"]

        if kind == "user":
            if item.get("is_initial"):
                title, content = _extract_initial(
                    item.get("text", ""),
                    item.get("title", ""),
                    item.get("content", ""),
                )
                title = _strip_emoji(title).strip()
                content = _clean_content(content)
                t.turns.append(Turn(kind="initial", body=content, title=title))
            else:
                t.turns.append(Turn(kind="user_r",
                                    body=_strip_emoji(item["text"]).strip()))

        elif kind == "moderator":
            q_body = _strip_emoji(item["text"]).strip()
            if pending_prompt_label:
                q_body = f"[{pending_prompt_label}]\n\n{q_body}"
                pending_prompt_label = None
            t.turns.append(Turn(kind="mod_q", body=q_body))

        elif kind == "prompt_label":
            pending_prompt_label = item["text"]

        elif kind == "moderator_thinking":
            body = _clean_internal(_strip_emoji(item["text"]))
            t.turns.append(Turn(kind="internal", body=body))

        elif kind == "verdict":
            result_text = "CORRECT" if item["correct"] else "INCORRECT"
            t.verdict = {
                "predicted":        item["predicted"],
                "ground_truth":     item["truth"],
                "predicted_intent": item["intent"],
                "true_intent":      item["true_intent"],
                "result":           result_text,
            }

    return t


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.") + "..."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLORS = {
    "page_bg":        "#ffffff",
    "surface_bg":     "#ffffff",
    "surface_border": (0, 0, 0, 0.10),
    "text_primary":   "#1a1a1a",
    "text_secondary": "#5f5e5a",
    "text_tertiary":  "#888780",
    "user_bubble":    "#ffffff",
    "user_border":    (0, 0, 0, 0.10),
    "user_initial_l": "#185FA5",
    "user_initial_b": (0, 0, 0, 0.18),
    "info_bg":        "#E6F1FB",
    "info_text":      "#0C447C",
    "mod_bg":         "#FAECE7",
    "mod_text":       "#4A1B0C",
    "mod_border":     "#F0997B",
    "mod_icon_text":  "#712B13",
    "internal_border":     (0, 0, 0, 0.18),
    "internal_final_brd":  (0, 0, 0, 0.30),
    "meta_pill_bg":   "#faf9f5",
}

# Page width chosen so PDF output is ~570pt wide (matching figure2.pdf).
#   figsize_in = FIG_WIDTH_PX / DPI,  pdf_width_pt = figsize_in * 72
#   1188 / 150 * 72 ≈ 570pt
FIG_WIDTH_PX = 1188
DPI = 150
# Asymmetric page padding: extra room on top and left, normal on right/bottom.
PAGE_PAD_TOP    = 54
PAGE_PAD_LEFT   = 54
PAGE_PAD_RIGHT  = 36
PAGE_PAD_BOTTOM = 36
SURFACE_PAD = 26
BUBBLE_PAD_X = 16
BUBBLE_PAD_Y_TOP = 20
BUBBLE_PAD_Y_BOTTOM = 14
INT_PAD_X = 14
INT_PAD_Y_TOP = 17
INT_PAD_Y_BOTTOM = 11

# Font sizes are in points and map 1:1 to PDF points, so these match
# figure2.pdf visually on the wider page.
FS_TITLE   = 10.5
FS_BODY    = 8.5
FS_LABEL   = 7.8
FS_EYEBROW = 6.8
FS_INT     = 8.0
FS_PILL    = 7.0


def pt_to_px(pt: float) -> float:
    return pt * DPI / 72.0


def char_width_px(fontsize_pt: float) -> float:
    return fontsize_pt * 0.55 * DPI / 72.0


def wrap_to_width(text: str, max_px: float, fontsize_pt: float) -> str:
    cw = char_width_px(fontsize_pt)
    chars = max(15, int(max_px / cw))
    paragraphs = text.split("\n\n")
    out = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        out.append(textwrap.fill(p, width=chars,
                                 break_long_words=False,
                                 break_on_hyphens=False))
    return "\n\n".join(out)


def text_height_px(text: str, fontsize_pt: float, linespacing: float = 1.55) -> float:
    if not text:
        return 0.0
    line_h = pt_to_px(fontsize_pt) * linespacing
    return (text.count("\n") + 1) * line_h


def add_rounded_box(ax, x, y, w, h, *, facecolor, edgecolor,
                    lw=0.5, linestyle="-", radius=8):
    r = min(radius, w / 2 - 0.5, h / 2 - 0.5)
    r = max(r, 0.1)
    box = FancyBboxPatch(
        (x + r, y + r),
        w - 2 * r, h - 2 * r,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        joinstyle="round",
        clip_on=False,
    )
    ax.add_patch(box)


def render(transcript: Transcript, out_stem: str,
           max_chars_initial: int = 700,
           max_chars_other: int = 500) -> Path:

    page_w = FIG_WIDTH_PX
    inner_w = page_w - PAGE_PAD_LEFT - PAGE_PAD_RIGHT - 2 * SURFACE_PAD
    BUBBLE_W_FULL = inner_w
    BUBBLE_W_MODQ = int(inner_w * 0.78)
    BUBBLE_W_INT  = int(inner_w * 0.70)

    layout = []

    def add_initial(turn):
        body = truncate(turn.body, max_chars_initial)
        wrap_w = BUBBLE_W_FULL - 2 * BUBBLE_PAD_X
        wrapped = wrap_to_width(body, wrap_w, FS_BODY)
        wrapped_title = wrap_to_width(turn.title, wrap_w, FS_TITLE) if turn.title else ""
        title_h = text_height_px(wrapped_title, FS_TITLE, 1.25) if wrapped_title else 0
        title_gap = 8 if wrapped_title else 0
        body_h = text_height_px(wrapped, FS_BODY)
        h = BUBBLE_PAD_Y_TOP + BUBBLE_PAD_Y_BOTTOM + title_h + title_gap + body_h
        layout.append(dict(
            kind="initial", label="User · initial post",
            community=transcript.community,
            title=wrapped_title, body=wrapped,
            w=BUBBLE_W_FULL, h=h, align="left",
        ))

    def add_user_r(turn, idx):
        body = truncate(turn.body, max_chars_other)
        wrap_w = BUBBLE_W_FULL - 2 * BUBBLE_PAD_X
        wrapped = wrap_to_width(body, wrap_w, FS_BODY)
        body_h = text_height_px(wrapped, FS_BODY)
        h = BUBBLE_PAD_Y_TOP + BUBBLE_PAD_Y_BOTTOM + body_h
        layout.append(dict(
            kind="user_r", label=f"User · response {idx}",
            body=wrapped, w=BUBBLE_W_FULL, h=h, align="left",
        ))

    def add_mod_q(turn, idx):
        wrap_w = BUBBLE_W_MODQ - 2 * BUBBLE_PAD_X
        wrapped = wrap_to_width(turn.body, wrap_w, FS_BODY)
        body_h = text_height_px(wrapped, FS_BODY)
        h = BUBBLE_PAD_Y_TOP + BUBBLE_PAD_Y_BOTTOM + body_h
        layout.append(dict(
            kind="mod_q", label=f"Moderator · question {idx}",
            body=wrapped, w=BUBBLE_W_MODQ, h=h, align="right",
        ))

    def add_internal(body, is_final=False):
        wrap_w = BUBBLE_W_INT - 2 * INT_PAD_X
        wrapped = wrap_to_width(body, wrap_w, FS_INT)
        body_h = text_height_px(wrapped, FS_INT)
        eyebrow_h = pt_to_px(FS_EYEBROW) * 1.4 + 4
        h = INT_PAD_Y_TOP + INT_PAD_Y_BOTTOM + eyebrow_h + body_h
        layout.append(dict(
            kind="internal_final" if is_final else "internal",
            label="Moderator · final verdict" if is_final else "Moderator · internal",
            body=wrapped, w=BUBBLE_W_INT, h=h, align="right",
        ))

    q_idx = r_idx = 0
    has_synth_final = bool(transcript.verdict)

    for turn in transcript.turns:
        if turn.kind == "initial":
            add_initial(turn)
        elif turn.kind == "internal":
            add_internal(turn.body, is_final=False)
        elif turn.kind == "mod_q":
            q_idx += 1
            add_mod_q(turn, q_idx)
        elif turn.kind == "user_r":
            r_idx += 1
            add_user_r(turn, r_idx)

    if has_synth_final:
        v = transcript.verdict
        pred = v.get("predicted", "").lower()
        intent = v.get("predicted_intent", "")
        sentence = (f"Sufficient evidence gathered. Classifying post as {pred} "
                    f"with intent {intent}.")
        add_internal(sentence, is_final=True)

    GAP = 16
    LABEL_H = 24
    LABEL_GAP = 6

    total_h = PAGE_PAD_TOP + SURFACE_PAD
    for it in layout:
        if it["kind"] in ("initial", "user_r", "mod_q"):
            total_h += LABEL_H + LABEL_GAP
        total_h += it["h"] + GAP
    total_h -= GAP
    total_h += SURFACE_PAD + PAGE_PAD_BOTTOM

    fig = plt.figure(figsize=(page_w / DPI, total_h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(total_h, 0)
    ax.set_axis_off()
    fig.patch.set_facecolor(COLORS["page_bg"])

    surface_x = PAGE_PAD_LEFT
    surface_y = PAGE_PAD_TOP
    surface_w = page_w - PAGE_PAD_LEFT - PAGE_PAD_RIGHT
    surface_h = total_h - PAGE_PAD_TOP - PAGE_PAD_BOTTOM
    add_rounded_box(ax, surface_x, surface_y, surface_w, surface_h,
                    facecolor=COLORS["surface_bg"],
                    edgecolor=COLORS["surface_border"],
                    lw=0.5, radius=12)

    def draw_avatar(cx, cy, kind):
        if kind == "user":
            bg, fg, letter = COLORS["info_bg"], COLORS["info_text"], "U"
        else:
            bg, fg, letter = COLORS["mod_bg"], COLORS["mod_icon_text"], "M"
        ax.add_patch(Circle((cx, cy), 10, facecolor=bg,
                            edgecolor="none", clip_on=False))
        ax.text(cx, cy, letter, ha="center", va="center",
                fontsize=7.2, fontweight="700", color=fg, clip_on=False)

    def measure_text_width(s: str, fontsize: float, weight: str = "normal",
                           family: str = "sans-serif") -> float:
        t = ax.text(0, 0, s, fontsize=fontsize, fontweight=weight,
                    family=family, alpha=0)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
        width_px = bbox.width
        t.remove()
        return width_px

    def draw_pill(x, y, text):
        text_w = measure_text_width(text, FS_PILL, family="monospace")
        w = text_w + 18
        h = 18
        r = h / 2
        bg = "#f1eee7"
        edge = (0, 0, 0, 0.28)
        ax.add_patch(Rectangle((x + r, y), w - 2 * r, h,
                               facecolor=bg, edgecolor="none", clip_on=False))
        ax.add_patch(Circle((x + r, y + r), r,
                            facecolor=bg, edgecolor="none", clip_on=False))
        ax.add_patch(Circle((x + w - r, y + r), r,
                            facecolor=bg, edgecolor="none", clip_on=False))
        from matplotlib.patches import Arc
        ax.plot([x + r, x + w - r], [y, y],
                color=edge, lw=0.6, solid_capstyle="butt", clip_on=False)
        ax.plot([x + r, x + w - r], [y + h, y + h],
                color=edge, lw=0.6, solid_capstyle="butt", clip_on=False)
        ax.add_patch(Arc((x + r, y + r), 2 * r, 2 * r,
                         theta1=90, theta2=270, color=edge, lw=0.6, clip_on=False))
        ax.add_patch(Arc((x + w - r, y + r), 2 * r, 2 * r,
                         theta1=-90, theta2=90, color=edge, lw=0.6, clip_on=False))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=FS_PILL, color=COLORS["text_secondary"],
                family="monospace", clip_on=False)

    cy = PAGE_PAD_TOP + SURFACE_PAD

    for it in layout:
        has_label = it["kind"] in ("initial", "user_r", "mod_q")
        align = it["align"]

        if has_label:
            label_y_center = cy + LABEL_H / 2
            if align == "left":
                avatar_x = surface_x + SURFACE_PAD + 10
                draw_avatar(avatar_x, label_y_center, "user")
                label_x = avatar_x + 10 + 8
                ax.text(label_x, label_y_center, it["label"],
                        ha="left", va="center",
                        fontsize=FS_LABEL, color=COLORS["text_secondary"],
                        fontweight="500", clip_on=False)
                if it["kind"] == "initial" and it.get("community"):
                    label_w = measure_text_width(it["label"], FS_LABEL, "500")
                    pill_x = label_x + label_w + 12
                    draw_pill(pill_x, label_y_center - 9, it["community"])
            else:
                right_edge = surface_x + surface_w - SURFACE_PAD
                avatar_x = right_edge - 10
                draw_avatar(avatar_x, label_y_center, "mod")
                label_x = avatar_x - 10 - 8
                ax.text(label_x, label_y_center, it["label"],
                        ha="right", va="center",
                        fontsize=FS_LABEL, color=COLORS["text_secondary"],
                        fontweight="500", clip_on=False)
            cy += LABEL_H + LABEL_GAP

        bx = surface_x + SURFACE_PAD if align == "left" \
            else surface_x + surface_w - SURFACE_PAD - it["w"]
        by = cy

        if it["kind"] == "initial":
            add_rounded_box(ax, bx, by, it["w"], it["h"],
                            facecolor=COLORS["user_bubble"],
                            edgecolor=COLORS["user_initial_b"],
                            lw=0.5, radius=8)
            ax.add_patch(Rectangle(
                (bx + 0.5, by + 4), 2.5, it["h"] - 8,
                facecolor=COLORS["user_initial_l"],
                edgecolor="none", clip_on=False))
        elif it["kind"] == "user_r":
            add_rounded_box(ax, bx, by, it["w"], it["h"],
                            facecolor=COLORS["user_bubble"],
                            edgecolor=COLORS["user_border"],
                            lw=0.5, radius=8)
        elif it["kind"] == "mod_q":
            add_rounded_box(ax, bx, by, it["w"], it["h"],
                            facecolor=COLORS["mod_bg"],
                            edgecolor=COLORS["mod_border"],
                            lw=0.5, radius=8)
        elif it["kind"] in ("internal", "internal_final"):
            is_final = it["kind"] == "internal_final"
            add_rounded_box(ax, bx, by, it["w"], it["h"],
                            facecolor=COLORS["surface_bg"],
                            edgecolor=(COLORS["internal_final_brd"] if is_final
                                       else COLORS["internal_border"]),
                            lw=0.6 if is_final else 0.5,
                            linestyle="-" if is_final else (0, (3, 3)),
                            radius=8)

        if it["kind"] in ("internal", "internal_final"):
            tx = bx + INT_PAD_X
            ty = by + INT_PAD_Y_TOP
            ax.text(tx, ty, it["label"].upper(),
                    ha="left", va="top",
                    fontsize=FS_EYEBROW, color=COLORS["text_tertiary"],
                    fontweight="700", clip_on=False)
            ty += pt_to_px(FS_EYEBROW) * 1.4 + 4
            ax.text(tx, ty, it["body"],
                    ha="left", va="top",
                    fontsize=FS_INT, color=COLORS["text_secondary"],
                    fontstyle="italic", linespacing=1.55, clip_on=False)
        else:
            tx = bx + BUBBLE_PAD_X
            ty = by + BUBBLE_PAD_Y_TOP
            if it["kind"] == "initial" and it.get("title"):
                ax.text(tx, ty, it["title"],
                        ha="left", va="top",
                        fontsize=FS_TITLE, color=COLORS["text_primary"],
                        fontweight="700", clip_on=False)
                ty += text_height_px(it["title"], FS_TITLE, 1.25) + 8
            color = COLORS["mod_text"] if it["kind"] == "mod_q" else COLORS["text_primary"]
            ax.text(tx, ty, it["body"],
                    ha="left", va="top",
                    fontsize=FS_BODY, color=color,
                    linespacing=1.55, clip_on=False)

        cy += it["h"] + GAP

    pdf = Path(out_stem).with_suffix(".pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, facecolor=COLORS["page_bg"])
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default=str(REPO_ROOT / "data/dataset/id/posts-test-generated-llama.json"),
                    help="Path to generated posts JSON")
    ap.add_argument("--start", type=int, default=10,
                    help="Zero-based index of first conversation to run")
    ap.add_argument("--count", type=int, default=20,
                    help="Number of conversations to run")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="Directory to write PDFs into")
    ap.add_argument("--max-initial", type=int, default=700)
    ap.add_argument("--max-other", type=int, default=500)
    args = ap.parse_args()

    print("Loading test data...")
    with open(args.data) as f:
        all_data = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i in range(args.start, args.start + args.count):
        print(f"\n{'='*80}")
        print(f"Processing conversation {i+1}/{args.start + args.count}")
        print(f"{'='*80}")

        trace, community, title = run_trace(all_data[i])
        transcript = trace_to_transcript(trace, community)

        pdf_path = render(
            transcript,
            str(args.out_dir / f"conversation_{i+1}"),
            max_chars_initial=args.max_initial,
            max_chars_other=args.max_other,
        )

        verdict = trace[-1]
        results.append({
            "pdf":       str(pdf_path),
            "title":     title,
            "community": community,
            "correct":   verdict["correct"],
            "predicted": verdict["predicted"],
            "truth":     verdict["truth"],
        })

        print(f"✓ Wrote {pdf_path}")

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = (correct / total * 100) if total else 0
    print(f"\n{'='*80}")
    print(f"Done — {total} conversations, {correct} correct ({accuracy:.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
