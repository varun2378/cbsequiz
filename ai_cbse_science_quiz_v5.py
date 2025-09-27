# ai_cbse_science_quiz.py
# CBSE Class 6 – AI-Generated Objective Quiz
# Subject → Primary → Subtopics (NO file upload option)
# -------------------------------------------------------------------------
# Folder layout:
#   C:\CBSEQuestionsGenerated_V2\
#       science_topics.txt
#       social_studies_topics.txt
#       ...
# Each *.txt file uses lines:
#   Primary=Sub1,Sub2,Sub3
#   Another Primary:SubA,SubB
#
# This app:
#   1) Shows a SUBJECT dropdown (files in the folder).
#   2) Shows PRIMARY topics from the selected file.
#   3) Generates questions across the PRIMARY's subtopics (even mix option).
#   4) MCQ + Match-the-Following + Short Answer (shorts always at end), explanations shown.
#   5) Weekly cache, per-candidate score saving + scorebook.
#   6) Short-answer fuzzy scoring (full/partial) + teacher override panel.
#   7) MCQ correct answer positions shuffled & balanced across A/B/C/D.
#  8) Matching short answers if partially correct. For e.g.  60-70% will give full marks

import os
import io
import json
import time
import random
import shutil
import hashlib
import re
import difflib
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

import streamlit as st
from dotenv import load_dotenv
from jsonschema import validate
from pydantic import BaseModel, Field, ValidationError, conint, field_validator, model_validator
import pandas as pd

# --------------------------- CONFIG ---------------------------

APP_ROOT = os.path.dirname(__file__)
SUBJECTS_DIR = r"C:\CBSEQuestionsGenerated_V2"  # scans for *.txt here
CACHE_DIR = os.path.join(APP_ROOT, "quiz_cache_v2")
SCORES_FILE = os.path.join(CACHE_DIR, "scores.json")
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week

LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"  # change if desired (e.g., "gpt-4o")
MAX_RETRIES = 3                # modest for responsiveness
REPAIR_TRIES_PER_QUESTION = 2  # try to repair a bad MCQ up to 2 times
CACHE_VERSION = "v8_mcq_balance"  # bump so old cache is ignored

# ---- Short-answer fuzzy scoring thresholds ----
SHORT_FULL_CREDIT_THRESHOLD = 0.60   # ≥60% → full 1.0 mark
SHORT_PARTIAL_MIN, SHORT_PARTIAL_MAX = 0.40, 0.59  # 40–59% → 0.5 mark

# ------------------------ UTIL: CACHE -------------------------

def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(subject: str, topics: List[str], n_q: int, weights: Dict[str, int], seed: int, even_mix: bool) -> str:
    payload = json.dumps({
        "cache_version": CACHE_VERSION,  # versioned cache
        "subject": subject,
        "topics": sorted(topics),
        "n": n_q,
        "weights": weights,
        "seed": seed,
        "even_mix": even_mix
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def is_cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age <= CACHE_TTL_SECONDS

def write_cache(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_cache(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def reset_cache() -> None:
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------ UTIL: SCORES ------------------------

def _ensure_scores_file() -> None:
    ensure_cache_dir()
    if not os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_scores() -> List[Dict[str, Any]]:
    _ensure_scores_file()
    try:
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_score(entry: Dict[str, Any]) -> None:
    _ensure_scores_file()
    data = load_scores()
    data.append(entry)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_scores() -> None:
    _ensure_scores_file()
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ----------------------- UTIL: SUBJECTS -----------------------

def list_subject_files(folder: str) -> Dict[str, str]:
    """
    Returns {display_name: full_path} for all *.txt files in the folder.
    The display name is the filename without extension (underscores -> spaces).
    """
    subjects = {}
    if not os.path.isdir(folder):
        return subjects
    for name in os.listdir(folder):
        if name.lower().endswith(".txt"):
            base = os.path.splitext(name)[0]
            display = base.replace("_", " ").strip()
            subjects[display] = os.path.join(folder, name)
    return subjects

def parse_topics_file(path: str) -> Dict[str, List[str]]:
    """
    topics file lines:
      Primary=Sub1,Sub2,Sub3
      Another Primary:SubA,SubB
    Returns {Primary: [Sub1, ...]}.
    """
    mapping: Dict[str, List[str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    primary, tail = line.split("=", 1)
                elif ":" in line:
                    primary, tail = line.split(":", 1)
                else:
                    mapping[line] = []
                    continue
                primary = primary.strip()
                subs = [s.strip() for s in (tail or "").split(",") if s.strip()]
                mapping[primary] = subs
    except FileNotFoundError:
        mapping = {}
    return mapping

# ----------------------- DATA MODELS --------------------------

class Choice(BaseModel):
    text: str = Field(..., min_length=1)

class Question(BaseModel):
    id: str
    type: Literal["mcq", "match", "short"] = "mcq"  # + short
    prompt: str = Field(..., min_length=5)
    difficulty: str = Field(..., pattern=r"^(simple|moderate|tough)$")

    # MCQ
    choices: Optional[List[Choice]] = None
    correct_index: Optional[conint(ge=0)] = None

    # Match-the-Following
    match_left: Optional[List[str]] = None
    match_right: Optional[List[str]] = None
    # match_answer[i] is the index in match_right that matches match_left[i]
    match_answer: Optional[List[int]] = None

    # Short Answer
    expected_answers: Optional[List[str]] = None

    explanation: str = Field(..., min_length=5)

    @field_validator("id", mode="before")
    def _coerce_id(cls, v):
        return str(v)

    @model_validator(mode="after")
    def _check_by_type(self):
        if self.type == "mcq":
            if not self.choices or self.correct_index is None:
                raise ValueError("MCQ requires 'choices' and 'correct_index'.")
            if len(self.choices) < 4:
                raise ValueError("MCQ must have at least 4 choices.")
            if not (0 <= self.correct_index < len(self.choices)):
                raise ValueError("MCQ 'correct_index' out of range.")
        elif self.type == "match":
            if not self.match_left or not self.match_right or self.match_answer is None:
                raise ValueError("Match requires 'match_left', 'match_right', 'match_answer'.")
            if len(self.match_left) != len(self.match_right) or len(self.match_left) != len(self.match_answer):
                raise ValueError("Match lists must be same length.")
            n = len(self.match_right)
            if any((not isinstance(i, int) or i < 0 or i >= n) for i in self.match_answer):
                raise ValueError("Match 'match_answer' indices invalid.")
        else:  # short
            if not self.expected_answers or not isinstance(self.expected_answers, list):
                raise ValueError("Short requires 'expected_answers' list (>=1).")
            if not all(isinstance(s, str) and s.strip() for s in self.expected_answers):
                raise ValueError("'expected_answers' must be non-empty strings.")
        return self

class QuizPayload(BaseModel):
    topic: str  # label we display (subject | primary | joined subtopics)
    questions: List[Question] = Field(..., min_items=1)

QUIZ_JSON_SCHEMA = {
    "type": "object",
    "required": ["topic", "questions"],
    "properties": {
        "topic": {"type": "string"},
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "oneOf": [
                    {   # MCQ
                        "type": "object",
                        "required": ["id", "type", "prompt", "difficulty", "choices", "correct_index", "explanation"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "const": "mcq"},
                            "prompt": {"type": "string"},
                            "difficulty": {"type": "string", "enum": ["simple", "moderate", "tough"]},
                            "choices": {
                                "type": "array",
                                "minItems": 4,
                                "items": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
                            },
                            "correct_index": {"type": "integer", "minimum": 0},
                            "explanation": {"type": "string"},
                        },
                        "additionalProperties": False
                    },
                    {   # Match-the-Following
                        "type": "object",
                        "required": ["id", "type", "prompt", "difficulty", "match_left", "match_right", "match_answer", "explanation"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "const": "match"},
                            "prompt": {"type": "string"},
                            "difficulty": {"type": "string", "enum": ["simple", "moderate", "tough"]},
                            "match_left": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                            "match_right": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                            "match_answer": {"type": "array", "minItems": 2, "items": {"type": "integer", "minimum": 0}},
                            "explanation": {"type": "string"},
                        },
                        "additionalProperties": False
                    },
                    {   # Short Answer
                        "type": "object",
                        "required": ["id", "type", "prompt", "difficulty", "expected_answers", "explanation"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "const": "short"},
                            "prompt": {"type": "string"},
                            "difficulty": {"type": "string", "enum": ["simple", "moderate", "tough"]},
                            "expected_answers": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                            "explanation": {"type": "string"},
                        },
                        "additionalProperties": False
                    }
                ]
            }
        }
    },
    "additionalProperties": False
}

# ------------------------ LLM LAYER ---------------------------

def generate_with_llm(subject: str, topics: List[str], n_questions: int, weights: Dict[str, int], seed: int, even_mix: bool) -> Dict[str, Any]:
    if LLM_PROVIDER == "openai":
        return _generate_with_openai(subject, topics, n_questions, weights, seed, even_mix)
    raise RuntimeError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def _is_placeholder(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return True
    if len(t) <= 3:
        return True
    if t in {"a", "b", "c", "d"}:
        return True
    if t in {"text", "choice", "select", "none"}:
        return True
    if re.fullmatch(r"option\s*\d+", t) or t.startswith("option "):
        return True
    if "all of the above" in t or "none of the above" in t:
        return True
    return False

def _heuristic_fill_choices(prompt: str) -> List[str]:
    """
    Last-resort non-placeholder choices if repairs fail.
    We make one positive/correct, and 3 plausible distractors.
    """
    base = "the concept described in the question"
    return [
        f"Supports {base}",
        f"Opposes {base}",
        f"Unrelated to {base}",
        f"Exaggerated statement about {base}",
    ]

def _repair_mcq_with_llm(client, subject: str, subtopics: List[str], seed: int, q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to repair ONLY the choices/correct_index for one MCQ.
    """
    system_msg = "You repair multiple-choice questions for CBSE Class 6. Return ONLY JSON."
    user_msg = f"""Subject: {subject}
Subtopics: {', '.join(subtopics)}
Prompt: {q.get('prompt','')}

Task: Provide 4 MEANINGFUL choices (array of objects with 'text') and a correct_index (0-3). 
Forbidden: 'A','B','C','D','text','choice','select','none','option 1/2/3/4','all/none of the above', empty or 1–3 letter strings.
Return JSON: {{"choices":[{{"text":""}},{{"text":""}},{{"text":""}},{{"text":""}}], "correct_index":0}}"""

    try:
        from openai import OpenAI
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            seed=seed,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        rc = data.get("choices", [])
        if isinstance(rc, list) and len(rc) >= 4:
            cleaned = []
            for c in rc[:4]:
                txt = c.get("text") if isinstance(c, dict) else c
                cleaned.append({"text": str(txt).strip()})
            ci = data.get("correct_index", 0)
            if not isinstance(ci, int) or ci < 0 or ci > 3:
                ci = 0
            q["choices"] = cleaned
            q["correct_index"] = ci
    except Exception:
        pass
    return q

# ---- MCQ shuffling & balancing helpers ----
def _rebalance_mcq_positions(qs: List[Dict[str, Any]], seed: int) -> None:
    """
    For each MCQ in order:
      1) Shuffle the 3 distractors.
      2) Place the correct choice into the least-used position among [0..3].
    Deterministic per seed; operates IN-PLACE.
    """
    counts = [0, 0, 0, 0]               # how many times each position used so far
    rnd = random.Random(int(seed) ^ 0x5EED)  # deterministic per seed, different stream

    for q in qs:
        if q.get("type") != "mcq":
            continue
        choices = q.get("choices") or []
        ci = q.get("correct_index", 0)
        if not choices or not isinstance(ci, int) or not (0 <= ci < len(choices)):
            continue

        # Ensure exactly 4 choices
        while len(choices) < 4:
            choices.append({"text": f"Choice {len(choices)+1}"})
        choices = choices[:4]

        correct = choices[ci]
        others = [c for i, c in enumerate(choices) if i != ci]
        rnd.shuffle(others)

        # pick least-used slot for the correct answer
        desired = counts.index(min(counts))
        new_choices = [None] * 4
        new_choices[desired] = correct

        # fill remaining slots with shuffled distractors
        slots = [0, 1, 2, 3]
        slots.remove(desired)
        for pos, c in zip(slots, others[:3]):  # use first 3 distractors
            new_choices[pos] = c

        # fill any remaining Nones (safety)
        for i in range(4):
            if new_choices[i] is None:
                new_choices[i] = {"text": f"Choice {i+1}"}

        q["choices"] = new_choices
        q["correct_index"] = desired
        counts[desired] += 1

def _generate_with_openai(subject: str, topics: List[str], n_questions: int, weights: Dict[str, int], seed: int, even_mix: bool) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.stop()
        raise RuntimeError("OPENAI_API_KEY not set. Use environment or .env file.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install openai >= 1.0: pip install openai") from e

    client = OpenAI()

    topics_list = topics if isinstance(topics, list) else [topics]
    label_topic = f"{subject} | " + " | ".join(topics_list)

    # difficulty split (simple/moderate/tough)
    total_w = max(1, weights.get("simple", 0) + weights.get("moderate", 0) + weights.get("tough", 0))
    simple_n = max(0, round(n_questions * weights.get("simple", 0) / total_w))
    moderate_n = max(0, round(n_questions * weights.get("moderate", 0) / total_w))
    tough_n = max(0, n_questions - simple_n - moderate_n)

    # Decide short + match questions; shorts must appear at the end in returned array
    short_n = max(1 if n_questions >= 6 else 0, round(0.20 * n_questions))
    mtf_n   = max(1 if n_questions >= 6 else 0, round(0.25 * n_questions))
    mcq_n   = max(0, n_questions - mtf_n - short_n)

    # Even per-subtopic distribution hint (optional)
    if even_mix and topics_list:
        base = n_questions // len(topics_list)
        rem = n_questions % len(topics_list)
        per_topic = {t: base + (1 if i < rem else 0) for i, t in enumerate(topics_list)}
    else:
        per_topic = {}

    system_msg = (
        "You are a CBSE Class 6 question generator."
        " Produce strictly valid JSON only."
        " Stay age-appropriate and follow NCERT Class 6 vocabulary."
    )
    user_instruction = f"""Generate {n_questions} objective questions for CBSE Class 6 **{subject}** covering these subtopics: {', '.join(topics_list)}.
Question types mix: ~{mcq_n} MCQ (type='mcq'), ~{mtf_n} Match (type='match'), and ~{short_n} Short Answer (type='short').
Difficulties across the whole set: simple={simple_n}, moderate={moderate_n}, tough={tough_n}.
{f"Distribute approximately per subtopic as: {per_topic}." if per_topic else "Mix naturally across subtopics."}

CRITICAL RULES for MCQs:
- The **stem** must be a clear, contentful sentence (≥ 25 characters).
- Each MCQ must have **4 meaningful** distinct choices. Choices must be self-contained phrases (e.g., "Improves digestion"), not labels or placeholders.
- Absolutely **forbidden** as choices: "A", "B", "C", "D", "text", "choice", "select", "none", "all of the above", "option 1/2/3/4" (case-insensitive), or any empty/1–3 character strings.

PLACEMENT:
- Put all 'short' questions at the **END** of the 'questions' array.

Return ONLY valid JSON with:
  topic: string
  questions: array of objects, each EITHER
    type='mcq'   with fields: id, prompt, difficulty, choices{{text}}, correct_index, explanation
    type='match' with fields: id, prompt, difficulty, match_left[str[]], match_right[str[]], match_answer[int[]], explanation
    type='short' with fields: id, prompt, difficulty, expected_answers[str[]], explanation

Rules:
- MCQ: exactly 4 choices; exactly one correct_index.
- Match: match_left and match_right same length; match_answer[i] is the index in match_right that matches match_left[i].
- Short: expected_answers should contain 2–5 acceptable variants (synonyms / phrasing).
- Use age-appropriate NCERT Class 6 language."""

    last_error = None
    data = None

    # ---- Try whole-batch generation a few times (no hard failure) ----
    for attempt in range(MAX_RETRIES):
        try:
            temp = [0.4, 0.6, 0.8][min(attempt, 2)]
            with st.spinner(f"Generating quiz (attempt {attempt+1}/{MAX_RETRIES})..."):
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_instruction},
                    ],
                    temperature=temp,
                    seed=seed,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content)
            break
        except Exception as e:
            last_error = e
            data = None
            continue

    if not data:
        raise RuntimeError(f"Quiz generation failed: {last_error}")

    # Ensure topic label for schema/UI
    data["topic"] = label_topic

    # --- SANITIZE, then soft-repair bad MCQs instead of failing ---
    qs = data.get("questions", [])
    for i, q in enumerate(qs):
        q["id"] = str(q.get("id", i + 1))
        q_type = str(q.get("type", "mcq")).lower().strip()
        if q_type not in ("mcq", "match", "short"):
            q_type = "mcq"
        q["type"] = q_type

        # difficulty normalize
        if q.get("difficulty") not in ("simple", "moderate", "tough"):
            q["difficulty"] = "moderate"

        if q_type == "mcq":
            # prompt
            prompt = str(q.get("prompt", "")).strip()
            q["prompt"] = prompt if len(prompt) >= 25 else "Read the question carefully and choose the correct option."

            # choices
            raw_choices = q.get("choices", [])
            norm = []
            for c in raw_choices:
                if isinstance(c, dict) and "text" in c:
                    txt = str(c["text"]).strip()
                else:
                    txt = str(c).strip()
                norm.append({"text": txt})
            while len(norm) < 4:
                norm.append({"text": f"Option {len(norm)+1}"})
            norm = norm[:4]

            # quality checks
            norm = [{"text": c["text"].strip()} for c in norm]
            placeholders = sum(1 for c in norm if _is_placeholder(c["text"]))
            unique_texts = {c["text"].lower() for c in norm}
            too_few_unique = len(unique_texts) < 4
            low_quality = placeholders > 0 or too_few_unique or len(q["prompt"]) < 25

            # correct index
            ci = q.get("correct_index", 0)
            if not isinstance(ci, int) or not (0 <= ci < 4):
                ci = 0

            q["choices"] = norm
            q["correct_index"] = ci
            q["_low_quality"] = low_quality

        elif q_type == "match":
            left = q.get("match_left") or []
            right = q.get("match_right") or []
            ans = q.get("match_answer")
            left = [str(x) for x in left]
            right = [str(x) for x in right]
            n = min(len(left), len(right))
            if n < 2:
                left, right, ans = ["A", "B"], ["1", "2"], [0, 1]
                n = 2
            else:
                left = left[:n]
                right = right[:n]
            if not isinstance(ans, list) or len(ans) != n or any((not isinstance(k, int) or k < 0 or k >= n) for k in ans):
                ans = list(range(n))
            q["match_left"] = left
            q["match_right"] = right
            q["match_answer"] = ans

            prompt = str(q.get("prompt", "")).strip()
            if len(prompt) < 5:
                prompt = "Match the items in the left column with the correct items on the right."
            q["prompt"] = prompt

        else:  # short
            exp = q.get("expected_answers")
            if not isinstance(exp, list):
                exp = [exp] if isinstance(exp, (str, int, float)) else []
            seen = set()
            cleaned = []
            for s in exp:
                s = str(s).strip()
                if s and s.lower() not in seen:
                    cleaned.append(s)
                    seen.add(s.lower())
            if not cleaned:
                cleaned = ["N/A"]
            q["expected_answers"] = cleaned

            prompt = str(q.get("prompt", "")).strip()
            if len(prompt) < 5:
                prompt = "Answer briefly."
            q["prompt"] = prompt

        if not isinstance(q.get("explanation"), str) or not q["explanation"].strip():
            q["explanation"] = "Explanation: The correct answer reflects the key concept."

    # --- Soft-repair pass for only bad MCQs (no hard failure) ---
    bad_idxs = [i for i, q in enumerate(qs) if q.get("type") == "mcq" and q.get("_low_quality")]
    if bad_idxs:
        with st.spinner(f"Repairing {len(bad_idxs)} low-quality MCQ(s)..."):
            for i in bad_idxs:
                q = qs[i]
                # Try LLM-based repair a couple of times
                for _ in range(REPAIR_TRIES_PER_QUESTION):
                    q = _repair_mcq_with_llm(client, subject, topics_list, seed, q)
                    # re-check quality
                    norm = [{"text": c["text"].strip()} for c in (q.get("choices") or [])]
                    if len(norm) < 4:
                        while len(norm) < 4:
                            norm.append({"text": f"Option {len(norm)+1}"})
                    placeholders = sum(1 for c in norm if _is_placeholder(c["text"]))
                    unique_texts = {c["text"].lower() for c in norm}
                    too_few_unique = len(unique_texts) < 4
                    if placeholders == 0 and not too_few_unique:
                        q["_low_quality"] = False
                        q["choices"] = norm[:4]
                        if not isinstance(q.get("correct_index"), int) or not (0 <= q["correct_index"] < 4):
                            q["correct_index"] = 0
                        break
                # If still low-quality after repairs, do heuristic local fill
                if q.get("_low_quality"):
                    filled = _heuristic_fill_choices(q.get("prompt", ""))
                    q["choices"] = [{"text": t} for t in filled[:4]]
                    q["correct_index"] = 0
                    q["_low_quality"] = False
                    if not q.get("explanation"):
                        q["explanation"] = "Explanation: The first option best matches the idea in the question."

    # Remove helper flags
    for q in qs:
        if "_low_quality" in q:
            q.pop("_low_quality", None)

    # ---- NEW: shuffle & balance MCQ correct positions deterministically ----
    _rebalance_mcq_positions(qs, seed)

    # Ensure short-answer questions appear at the end
    data["questions"] = [q for q in qs if q.get("type") != "short"] + [q for q in qs if q.get("type") == "short"]

    # Validate (should pass after repairs/fill)
    validate(instance=data, schema=QUIZ_JSON_SCHEMA)
    QuizPayload(**data)
    return data

# ---------------------- SHORT-ANSWER FUZZY SCORING --------------

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = " ".join(str(s).split()).strip().casefold()
    s = re.sub(r"[^\w\s]", "", s)  # keep words/digits/spaces
    return s

def _seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def best_short_similarity(user_answer: str, expected_list: List[str]) -> (float, str):
    """
    Returns (best_similarity, best_expected). Similarity is blended:
      sim = 0.6 * SequenceMatcher + 0.4 * token-Jaccard
    """
    u = _normalize_text(user_answer)
    best_sim, best_ref = 0.0, ""
    for ref in expected_list or []:
        r = _normalize_text(ref)
        sim = 0.6 * _seq_ratio(u, r) + 0.4 * _token_jaccard(u, r)
        if sim > best_sim:
            best_sim, best_ref = sim, ref
    return best_sim, best_ref

def marks_from_similarity(sim: float) -> float:
    """Map similarity→marks using thresholds above."""
    if sim >= SHORT_FULL_CREDIT_THRESHOLD:
        return 1.0
    if SHORT_PARTIAL_MIN <= sim <= SHORT_PARTIAL_MAX:
        return 0.5
    return 0.0

# ---------------------- RENDERING & SCORING --------------------

def slugify(text: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in text)
    s = "-".join(filter(None, s.split("-")))
    return s or "topic"

def coerce_quiz(data: Dict[str, Any]) -> QuizPayload:
    return QuizPayload(**data)

# --------------------------- UI -------------------------------
def render_quiz(quiz: QuizPayload):
    # Keep any existing review context after submit
    review_ctx = st.session_state.get("review_ctx")

    # ------------------------ MAIN QUIZ FORM ------------------------
    with st.form("quiz_form"):
        for idx, q in enumerate(quiz.questions):
            with st.container(border=True):
                st.markdown(f"**Q{idx+1}. {q.prompt}**")
                tlabel = "MCQ" if q.type == "mcq" else ("Match the Following" if q.type == "match" else "Short Answer")
                st.caption(f"Type: {tlabel} • Difficulty: {q.difficulty.title()}")

                if q.type == "mcq":
                    options = [c.text for c in q.choices]
                    key = f"q_{q.id}_{idx}"
                    st.radio(
                        "Select one",
                        options=options,
                        index=None,
                        key=key,
                        label_visibility="collapsed",
                    )
                elif q.type == "match":
                    left = q.match_left
                    right = q.match_right
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown("**Left (A, B, C, ...)**")
                        for i, item in enumerate(left):
                            st.markdown(f"**{chr(65+i)}.** {item}")
                    with c2:
                        st.markdown("**Match to (choose right item)**")
                        for i, _ in enumerate(left):
                            key = f"m_{q.id}_{idx}_{i}"
                            st.selectbox(
                                f"Match for {chr(65+i)}",
                                options=right,
                                index=None,
                                key=key,
                                label_visibility="collapsed",
                                placeholder="Select",
                            )
                else:  # short
                    key = f"s_{q.id}_{idx}"
                    st.text_input("Your answer:", key=key, placeholder="Type your short answer here")

        submitted = st.form_submit_button("Submit Answers")

    # ------------------- ON SUBMIT: SCORE + BUILD REVIEW CTX -------------------
    if submitted:
        candidate = st.session_state.get("candidate_name_input", "").strip()
        if not candidate:
            st.warning("Please enter the candidate name (left sidebar) before submitting to record the score.")

        score = 0.0
        total = len(quiz.questions)
        details = []
        short_idxs = []

        def _norm(s):
            if s is None:
                return ""
            return " ".join(str(s).split()).strip().casefold()

        for idx, q in enumerate(quiz.questions):
            if q.type == "mcq":
                key = f"q_{q.id}_{idx}"
                chosen = st.session_state.get(key)
                correct_text = q.choices[q.correct_index].text
                is_correct = (_norm(chosen) == _norm(correct_text))
                score += 1.0 if is_correct else 0.0
                details.append({
                    "Q#": idx + 1,
                    "Type": "MCQ",
                    "Prompt": q.prompt,
                    "Your Answer": chosen or "—",
                    "Correct Answer": correct_text,
                    "Explanation": q.explanation,
                    "AutoCorrect": is_correct,   # for MCQ/Match
                })

            elif q.type == "match":
                left = q.match_left
                right = q.match_right
                user_text_map = []
                for i, _ in enumerate(left):
                    key = f"m_{q.id}_{idx}_{i}"
                    chosen_text = st.session_state.get(key)
                    user_text_map.append(chosen_text or "—")
                expected_text_map = []
                for i, j in enumerate(q.match_answer):
                    expected_text_map.append(right[j] if 0 <= j < len(right) else "?")
                is_correct = all(_norm(user_text_map[i]) == _norm(expected_text_map[i]) for i in range(len(left)))
                score += 1.0 if is_correct else 0.0
                details.append({
                    "Q#": idx + 1,
                    "Type": "Match",
                    "Prompt": q.prompt,
                    "Your Answer": "; ".join(f"{chr(65+i)}→{user_text_map[i]}" for i in range(len(user_text_map))),
                    "Correct Answer": "; ".join(f"{chr(65+i)}→{expected_text_map[i]}" for i in range(len(expected_text_map))),
                    "Explanation": q.explanation,
                    "AutoCorrect": is_correct,
                })

            else:  # short
                key = f"s_{q.id}_{idx}"
                user_ans = st.session_state.get(key, "")
                best_sim, best_ref = best_short_similarity(user_ans, q.expected_answers or [])
                auto_marks = marks_from_similarity(best_sim)
                score += auto_marks
                details.append({
                    "Q#": idx + 1,
                    "Type": "Short",
                    "Prompt": q.prompt,
                    "Your Answer": user_ans or "—",
                    "Correct Answer": "; ".join(q.expected_answers or []),
                    "Explanation": q.explanation,
                    "AutoMarks": auto_marks,                 # 0.0 / 0.5 / 1.0
                    "AutoSimilarity": round(best_sim, 3),    # 0..1
                    "AutoBestRef": best_ref,
                })
                short_idxs.append(idx)

        st.success(f"Total Score: {round(score, 2)} / {total} (1 mark per question max)")
        with st.expander("See detailed answers & explanations"):
            for d in details:
                with st.container(border=True):
                    st.markdown(f"**Q{d['Q#']}. {d['Prompt']}**")
                    st.markdown(f"Your Answer: {d['Your Answer']}")
                    st.markdown(f"**Correct Answer:** {d['Correct Answer']}")
                    st.markdown(f"_Explanation:_ {d['Explanation']}")
                    if d["Type"] == "Short":
                        st.markdown(f"Auto: **{d['AutoMarks']}** mark(s) — Similarity: {int(d['AutoSimilarity']*100)}% (best ref: _{d['AutoBestRef']}_)")
                    else:
                        st.markdown("✅ Correct" if d.get("AutoCorrect") else "❌ Incorrect")

        # Save auto-scored entry immediately
        try:
            entry = {
                "candidate": candidate or "(unnamed)",
                "topics": quiz.topic,
                "score": round(score, 2),
                "total": total,
                "percentage": round((score / total) * 100, 2) if total else 0.0,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "adjusted": False,
            }
            save_score(entry)
            st.info("Score saved.")
        except Exception as e:
            st.error(f"Could not save score: {e}")

        # Persist a REVIEW CONTEXT so the panel stays after reruns
        st.session_state["review_ctx"] = {
            "candidate": candidate or "(unnamed)",
            "topic": quiz.topic,
            "auto_score": round(score, 2),
            "total": total,
            "entry_timestamp": entry["timestamp"],
            "details": details,
            "short_idxs": short_idxs
        }
        review_ctx = st.session_state["review_ctx"]

    # ------------------- PERSISTENT SHORT-ANSWER REVIEW PANEL -------------------
    if review_ctx and review_ctx.get("short_idxs"):
        st.subheader("📝 Review & adjust short answers")
        st.caption("Auto-scored with fuzzy matching. You can override each short answer to 0 / 0.5 / 1.0 and save an adjusted score.")

        details = review_ctx["details"]
        short_idxs = review_ctx["short_idxs"]
        total = review_ctx["total"]
        candidate = review_ctx["candidate"]
        topic = review_ctx["topic"]

        # Initialize override keys once (sticky across reruns)
        for idx in short_idxs:
            d = details[idx]
            key = f"adj_short_mark_q{d['Q#']}"
            if key not in st.session_state:
                st.session_state[key] = float(d.get("AutoMarks", 0.0))

        # Render per-question controls with similarity context
        for idx in short_idxs:
            d = details[idx]
            key = f"adj_short_mark_q{d['Q#']}"
            colL, colR = st.columns([3, 2])
            with colL:
                st.markdown(f"**Q{d['Q#']}** — Auto: {d.get('AutoMarks',0.0)} mark(s) | "
                            f"Similarity: {int((d.get('AutoSimilarity',0.0))*100)}% "
                            f"| Best ref: _{d.get('AutoBestRef','')}_")
                st.markdown(f"Prompt: {d['Prompt']}")
                st.markdown(f"Your answer: **{d['Your Answer']}**")
            with colR:
                st.selectbox(
                    "Override mark",
                    options=[0.0, 0.5, 1.0],
                    index=[0.0, 0.5, 1.0].index(float(st.session_state[key])),
                    key=key,
                    help="Choose manual mark for this short answer"
                )
            st.divider()

        colA, colB = st.columns(2)
        with colA:
            if st.button("Apply short-answer adjustments", key="btn_apply_adjustments"):
                try:
                    # Sum all non-short auto marks (MCQ+Match are 0/1 you already counted)
                    auto_non_short = 0.0
                    manual_short_sum = 0.0

                    for d in details:
                        if d["Type"] == "Short":
                            manual_short_sum += float(st.session_state.get(f"adj_short_mark_q{d['Q#']}", float(d.get("AutoMarks",0.0))))
                        else:
                            auto_non_short += 1.0 if d.get("AutoCorrect", False) else 0.0

                    adjusted = auto_non_short + manual_short_sum

                    st.success(f"Adjusted Score: {round(adjusted,2)} / {total}")

                    adjustments_payload = []
                    for idx in short_idxs:
                        d = details[idx]
                        adjustments_payload.append({
                            "Q#": d["Q#"],
                            "auto_marks": float(d.get("AutoMarks", 0.0)),
                            "auto_similarity": float(d.get("AutoSimilarity", 0.0)),
                            "manual_marks": float(st.session_state.get(f"adj_short_mark_q{d['Q#']}", float(d.get("AutoMarks",0.0)))),
                            "prompt": d["Prompt"],
                            "your_answer": d["Your Answer"],
                            "expected": d["Correct Answer"],
                        })

                    adjusted_entry = {
                        "candidate": candidate,
                        "topics": topic,
                        "score": round(adjusted, 2),
                        "total": total,
                        "percentage": round((adjusted / total) * 100, 2) if total else 0.0,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "adjusted": True,
                        "adjusted_from_timestamp": review_ctx.get("entry_timestamp"),
                        "short_answer_adjustments": adjustments_payload,
                    }
                    save_score(adjusted_entry)
                    st.info("Adjusted score saved.")
                except Exception as e:
                    st.error(f"Could not save adjusted score: {e}")

        with colB:
            if st.button("Finish review (hide panel)", key="btn_finish_review"):
                # Clear the review context and the override selections
                for idx in short_idxs:
                    d = details[idx]
                    k = f"adj_short_mark_q{d['Q#']}"
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.pop("review_ctx", None)
                st.success("Review finished.")
                st.rerun()

    # ---------------------- CLEAR SELECTIONS BUTTON ----------------------
    if st.button("Clear selections", key="btn_clear_selections"):
        # Clear only widgets for current quiz
        for idx, q in enumerate(quiz.questions):
            if q.type == "mcq":
                key = f"q_{q.id}_{idx}"
                if key in st.session_state:
                    del st.session_state[key]
            elif q.type == "match":
                for i, _ in enumerate(q.match_left):
                    key = f"m_{q.id}_{idx}_{i}"
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                key = f"s_{q.id}_{idx}"
                if key in st.session_state:
                    del st.session_state[key]
        # also clear any lingering review (optional)
        if "review_ctx" in st.session_state:
            st.session_state.pop("review_ctx", None)
        st.rerun()

def main():
    ensure_cache_dir()
    st.set_page_config(page_title="CBSE Class 6 – AI Quiz", page_icon="📚", layout="wide")
    st.title("📚 CBSE Class 6 – AI-Generated Objective Quiz")
    st.caption("Pick a SUBJECT file → choose a PRIMARY topic → get questions across its SUBTOPICS (MCQ + Match + Short).")

    # Sidebar (always render)
    with st.sidebar:
        st.header("Settings")

        # Candidate name
        if "candidate_name_input" not in st.session_state:
            st.session_state["candidate_name_input"] = ""
        st.text_input("Candidate name", key="candidate_name_input")

        # Reset cache
        if st.button("Reset Cache (Delete all cached quizzes)", key="btn_reset_cache_sidebar"):
            reset_cache()
            st.success("Cache cleared.")

        # Scores section
        st.subheader("Scores")
        show_scores = st.checkbox("Show all test scores", value=False, key="chk_show_scores")
        if st.button("Clear all scores", key="btn_clear_scores"):
            clear_scores()
            st.success("All scores cleared.")

        # Difficulty weights
        st.subheader("Difficulty Mix")
        w_simple = st.slider("Simple", 0, 100, 40, key="w_simple")
        w_moderate = st.slider("Moderate", 0, 100, 40, key="w_moderate")
        w_tough = st.slider("Tough", 0, 100, 20, key="w_tough")

        seed_val = st.number_input("Random Seed (0=random)", min_value=0, value=0, step=1, key="seed_input")
        seed = seed_val if seed_val else random.randint(1, 10_000)

        st.divider()
        st.markdown("**LLM Provider:** OpenAI")
        st.markdown(f"Model: `{OPENAI_MODEL}`")

    # SUBJECTS: scan folder for *.txt
    subjects = list_subject_files(SUBJECTS_DIR)
    if not subjects:
        st.warning(f"No subject topic files (*.txt) found in: {SUBJECTS_DIR}")
        return

    subject_names = sorted(subjects.keys())
    subject_selected = st.selectbox("Choose a SUBJECT (topics file)", options=subject_names, index=0, key="subject_dropdown")

    # Load primary → subtopics mapping for the chosen subject file
    primary_map = parse_topics_file(subjects[subject_selected])
    subject_label = subject_selected

    primary_list = list(primary_map.keys())
    if not primary_list:
        st.warning("No primary topics found in the chosen subject file.")
        return

    colA, colB = st.columns([2, 1])
    with colA:
        primary_selected = st.selectbox("Choose a PRIMARY Topic", options=primary_list, index=0, key="primary_dropdown")
        subtopics_for_primary = primary_map.get(primary_selected, [])
        n_questions = st.slider("Number of Questions", 5, 30, 12, key="q_count_slider")
        even_mix = st.checkbox("Even mix across subtopics", value=True, key="even_mix_chk")
    with colB:
        st.write("")
        st.write("")
        generate_btn = st.button("Generate / Load from Cache", type="primary", key="btn_generate")

    weights = {"simple": w_simple, "moderate": w_moderate, "tough": w_tough}
    topics_for_llm = subtopics_for_primary if subtopics_for_primary else [primary_selected]

    # Generate or load
    if generate_btn and topics_for_llm:
        key_ = cache_key(subject_label, topics_for_llm, n_questions, weights, int(seed), even_mix)
        path = cache_path(key_)
        if is_cache_fresh(path):
            data = read_cache(path)
        else:
            try:
                data = generate_with_llm(subject_label, topics_for_llm, n_questions, weights, int(seed), even_mix)
                write_cache(path, data)
            except Exception as e:
                st.error(str(e))
                data = None
        st.session_state.quiz_data = data

    # Render quiz
    if st.session_state.get("quiz_data"):
        try:
            quiz = coerce_quiz(st.session_state["quiz_data"])
            render_quiz(quiz)
        except ValidationError as ve:
            st.error(f"Generated data failed validation: {ve}")

    # Scores table (optional)
    if st.session_state.get("chk_show_scores"):
        scores = load_scores()
        st.subheader("All Test Scores")
        if scores:
            df = pd.DataFrame(scores)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download scores (CSV)", csv, file_name="all_scores.csv", mime="text/csv", key="btn_dl_scores")
        else:
            st.info("No scores saved yet.")

    st.divider()
    st.markdown("**Marking Scheme:** MCQ/Match = 1 mark if correct; Short = fuzzy scoring with partial credit. Explanations and correct answers are shown after submission.")
    st.caption("Flow: Subject file → Primary → Subtopics. Enter candidate name to record marks; view/export all scores from the sidebar.")

if __name__ == "__main__":
    main()
