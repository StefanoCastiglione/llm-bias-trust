"""
Atomic claim decomposition pipeline with LLM helpers.

The module splits a sentence into atomic claims in two ways.
The first way uses a propositionizer model followed by rule-based steps: coordinated lists are split into one claim per item, words fused without a space are repaired, and claims whose main subject is a light demonstrative, such as this process, are merged into the previous claim.
The second way asks an LLM to decompose the sentence directly.
"""

from __future__ import annotations

import json
import re

import torch
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# MODEL LOADERS

def load_models(device: str = "cpu") -> dict:
    """Load spaCy and the propositionizer model.
    Return a dict with the keys nlp, prop_tokenizer, prop_model and device."""
    nlp = spacy.load("en_core_web_trf")

    prop_model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    prop_tokenizer = AutoTokenizer.from_pretrained(prop_model_name)
    prop_model = AutoModelForSeq2SeqLM.from_pretrained(prop_model_name).to(device)
    prop_model.eval()

    return {
        "nlp": nlp,
        "prop_tokenizer": prop_tokenizer,
        "prop_model": prop_model,
        "device": device,
    }


# PROPOSITIONIZER AND FIXES

def propositionize(content: str, models: dict, title: str = "", section: str = "") -> list[str]:
    tokenizer = models["prop_tokenizer"]
    model = models["prop_model"]
    device = models["device"]

    # model input made of title, section, content
    prompt = f"Title: {title}. Section: {section}. Content: {content}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    # the model should output a JSON list of claims
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fall back to the raw output as a single claim
        print(f"Invalid JSON output: {raw!r}")
        return [raw]

DEMONSTRATIVES = {"this", "that", "these", "those"}

# abstract nouns that usually refer back to an earlier statement
# they almost always need an explicit antecedent
SHELL_NOUNS = {
    "process", "property", "phenomenon", "mechanism", "pattern", "effect",
    "climate", "system", "structure", "method", "principle", "condition",
    "behavior", "behaviour", "characteristic", "feature", "quality", "factor",
    "element", "aspect", "issue", "fact", "reason", "result", "outcome",
    "situation", "circumstance", "approach", "technique", "phase", "stage",
    "trend", "tendency", "dynamic", "interaction",
}


def _dangling_demonstrative_subject(doc):
    """Return the subject token when it is a light demonstrative.
    This covers a bare demonstrative pronoun, a demonstrative determiner with a noun that has no adjective, compound, possessive or numeral modifier, and a demonstrative determiner with a shell noun that has at most one descriptor.
    Return None in every other case."""
    # the first subject attached to the root is the main subject
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None
    subjects = [t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")]
    if not subjects:
        return None
    subj = subjects[0]

    # case one is a bare demonstrative pronoun
    if subj.pos_ == "PRON" and subj.lemma_.lower() in DEMONSTRATIVES:
        return subj

    # case two is a noun with a demonstrative determiner
    if subj.pos_ in ("NOUN", "PROPN"):
        dems = [c for c in subj.children if c.dep_ == "det" and c.lemma_.lower() in DEMONSTRATIVES]
        if not dems:
            return None
        # possessives and numerals make the subject specific enough to stand alone
        hard_mods = [c for c in subj.children if c.dep_ in ("poss", "nummod")]
        if hard_mods:
            return None
        descriptors = [c for c in subj.children if c.dep_ in ("amod", "compound")]
        # no descriptor means a light subject
        if not descriptors:
            return subj
        # one descriptor is accepted only for shell nouns
        if len(descriptors) == 1 and subj.lemma_.lower() in SHELL_NOUNS:
            return subj
    return None


def merge_dangling_demonstrative_pass(claims: list[str], nlp) -> list[str]:
    """Go through the claims of one decomposition.
    Merge each claim with a light demonstrative subject into the claim before it.
    Claims are joined with a semicolon and the first letter of the merged claim is lowercased."""
    # an empty list has nothing to merge
    if not claims:
        return claims
    # the first claim is kept and each later claim is tested
    out = [claims[0]]
    for claim in claims[1:]:
        doc = nlp(claim)
        subj = _dangling_demonstrative_subject(doc)
        if subj is not None and out:
            # join to the previous claim after stripping trailing periods, spaces and commas
            prev = out[-1].rstrip(". ").rstrip(",")
            rest = claim.strip()
            # lowercase the first letter of the second claim
            merged = f"{prev}; {rest[0].lower() + rest[1:] if rest else rest}"
            out[-1] = merged
        else:
            # the claim stays separate
            out.append(claim)
    return out


def expand_coordinate_lists(claim: str, nlp) -> list[str]:
    """Split a coordinated list in an object, prepositional object, attribute or apposition into one claim per item.
    Premodifiers of the first item are copied to each other item that has none of its own.
    A list governed by between or among is skipped because it forms one unit.
    Only the first list found is expanded."""
    doc = nlp(claim)

    def render_tokens(toks):
        out = []
        for t in toks:
            out.append(t.text)
            out.append(t.whitespace_)
        return "".join(out).strip()

    for token in doc:
        # only objects, prepositional objects, attributes and appositions can head a list
        if token.dep_ not in ("dobj", "obj", "attr", "pobj", "appos"):
            continue

        # a pair governed by between or among is one unit
        if token.head.pos_ == "ADP" and token.head.lemma_.lower() in ("between", "among"):
            continue

        # the head and its conjuncts are the items in sentence order
        items = sorted([token, *token.conjuncts], key=lambda t: t.i)
        # a single item is not a list
        if len(items) < 2:
            continue

        # item tokens from its subtree without conjunctions, punctuation and conjunct branches
        def item_tokens(item):
            toks = []
            def visit(t):
                toks.append(t)
                for child in t.children:
                    if child.dep_ in ("cc", "punct", "conj"):
                        continue
                    visit(child)
            visit(item)
            return sorted(toks, key=lambda x: x.i)

        item_toks_list = [item_tokens(it) for it in items]

        def own_premodifiers(item, own_toks):
            return [t for t in own_toks if t.i < item.i]

        # items without own premodifiers inherit those of the first item
        first_premod = own_premodifiers(items[0], item_toks_list[0])
        if first_premod:
            for idx in range(1, len(items)):
                own_premod = own_premodifiers(items[idx], item_toks_list[idx])
                if not own_premod:
                    existing_indices = {t.i for t in item_toks_list[idx]}
                    to_add = [t for t in first_premod if t.i not in existing_indices]
                    item_toks_list[idx] = sorted(to_add + item_toks_list[idx], key=lambda t: t.i)

        # text before and after the list is shared by every new claim
        all_indices = [t.i for toks in item_toks_list for t in toks]
        list_start = min(all_indices)
        list_end = max(all_indices)

        prefix = doc[:list_start].text
        suffix = doc[list_end + 1:].text

        # one claim per item
        results = []
        for toks in item_toks_list:
            item_str = render_tokens(toks)
            new_claim = f"{prefix} {item_str} {suffix}".strip()
            # remove the space before , . ; : and ?
            new_claim = re.sub(r"\s+([,.;:?])", r"\1", new_claim)
            # collapse repeated spaces
            new_claim = re.sub(r"\s+", " ", new_claim).strip()
            # make sure every claim ends with final punctuation
            if not new_claim.endswith((".", "?", "!")):
                new_claim += "."
            results.append(new_claim)

        # only the first list found is expanded
        return results

    # no list found so the claim is unchanged
    return [claim]


def fix_fused_words(claim: str, original_text: str) -> str:
    """Insert a space in a claim word that is missing from the original text but equals two original words joined.
    The first word must appear before the second in the original, and they need not be next to each other.
    When several pairs fit, the pair with the smallest gap is used.
    Words are runs of letters and apostrophes, words under 4 letters are skipped and each part needs at least 2 letters.
    The replacement keeps the spelling of the original words."""
    orig_words = re.findall(r"[A-Za-z']+", original_text)
    orig_words_lower = [w.lower() for w in orig_words]
    orig_set = set(orig_words_lower)

    claim_words = re.findall(r"[A-Za-z']+", claim)
    fixed = claim
    for w in claim_words:
        wl = w.lower()
        # known words and very short words are skipped
        if wl in orig_set or len(wl) < 4:
            continue
        best = None  # gap and the joined pair of words
        # try each split leaving at least 2 letters on each side
        for i in range(2, len(wl) - 1):
            a, b = wl[:i], wl[i:]
            # both parts must be original words
            if a not in orig_set or b not in orig_set:
                continue
            a_positions = [k for k, ow in enumerate(orig_words_lower) if ow == a]
            b_positions = [k for k, ow in enumerate(orig_words_lower) if ow == b]
            for ap in a_positions:
                for bp in b_positions:
                    # the first part must come before the second
                    if bp > ap:
                        gap = bp - ap
                        # keep the closest pair
                        if best is None or gap < best[0]:
                            best = (gap, f"{orig_words[ap]} {orig_words[bp]}")
        # replace only the first occurrence
        if best:
            fixed = re.sub(rf"\b{re.escape(w)}\b", best[1], fixed, count=1)
    return fixed


def propositionize_and_expand(content: str, models: dict, expand_lists: bool = True) -> list[str]:
    """Split content into claims with the propositionizer.
    Repair fused words, split coordinated lists when expand_lists is true, repair fused words again and merge dangling demonstrative claims.
    Return the final list of claims."""
    nlp = models["nlp"]
    raw = propositionize(content, models)
    raw = [fix_fused_words(c, content) for c in raw]

    # coordinated lists are split only when requested
    if expand_lists:
        expanded = []
        for c in raw:
            expanded.extend(expand_coordinate_lists(c, nlp))
    else:
        expanded = raw

    # second repair pass on the expanded claims
    expanded = [fix_fused_words(c, content) for c in expanded]

    return merge_dangling_demonstrative_pass(expanded, nlp)


# LLM DECOMPOSITION

def llm_decomposer(
    original_text: str,
    base_url: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.0,
) -> list[str]:
    """Decompose a sentence into atomic claims with an LLM only, without the rule-based pipeline.
    Return an empty list when the API call or the JSON parsing fails."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    system_prompt = (
        "You are an expert fact-extraction system. Your task is to decompose a sentence "
        "into a list of atomic claims.\n\n"
        "Rules:\n"
        "1. ATOMICITY: Each claim must express exactly one fact. "
        "If the original sentence contains a list (e.g. 'risks: addiction, drowsiness, relapse'), "
        "generate one separate claim per list item, never group them.\n"
        "2. COVERAGE: Together, the claims must cover all factual content of the original sentence. "
        "Do not omit any information, including modifiers, quantities, dates, locations, and qualifications.\n"
        "3. STRUCTURE: Each claim must be a complete, grammatically correct sentence "
        "with an explicit subject, verb, and object or complement. "
        "Never use pronouns without a clear antecedent: always use the full noun phrase.\n"
        "4. FAITHFULNESS: Do not introduce any information not present in the original sentence. "
        "Do not infer, generalize, or add context.\n"
        "5. SELF-CONTAINEDNESS: Each claim must be independently understandable without reading "
        "the other claims or the original sentence.\n"
        "6. NO NESTING: Do not produce claims that contain sub-clauses expressing separate facts. "
        "If a relative clause or apposition expresses an additional fact, extract it as a separate claim.\n\n"
        "CRITICAL: Return ONLY a JSON object with a single key 'claims' containing a list of strings. "
        "No markdown, no explanations."
    )
    user_content = json.dumps({"sentence": original_text}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        # the model must return a JSON object with the key claims
        parsed = json.loads(resp.choices[0].message.content.strip())
        claims = parsed.get("claims", [])
        # empty claims are dropped
        return [c.strip() for c in claims if c.strip()]
    except Exception as e:
        # on any error return no claims
        print(f"Decomposer failed: {e}")
        return []


# EXAMPLE

if __name__ == "__main__":
    BASE_URL = "https://your-endpoint/v1"
    API_KEY = "your-api-key"
    MODEL_NAME = "your-model"

    text = "Marie Curie discovered polonium and radium, and she won two Nobel Prizes."

    # loads spaCy and the propositionizer
    models = load_models()

    # decomposition into atomic claims
    claims = propositionize_and_expand(text, models)

    # decomposition into atomic claims with an LLM only
    llm_claims = llm_decomposer(text, BASE_URL, API_KEY, MODEL_NAME)
    print(llm_claims)
