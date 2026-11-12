"""
Multi-LLM claim aggregation with entailment matrices and an adapted TruthFinder algorithm.

The module provides the following functions.
build_entailment_matrix builds a signed matrix between claims from a semantic similarity filter, an NLI model, an optional LLM judge and an incoherence filter.
truthfinder computes the trustworthiness of each source and the confidence of each claim from that matrix.
flag_unsupported_claims, plot_truthfinder_history and plot_entailment_graph support diagnosis and visualization.

Only claims from different source models are compared, and claims from the same source model are never compared with each other.
"""

import math
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_call_utils import chat_completion_text

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # plain iterator when tqdm is missing
        return iterable


# DATA STRUCTURES

@dataclass
class AtomicClaim:
    claim_id: int
    text: str
    source_model: str


def claims_from_decomposition(
    claims_by_model: Dict[str, List[str]],
) -> List[AtomicClaim]:
    """
    Convert claim strings grouped by source model into AtomicClaim objects.

    Argument:
        claims_by_model: dict that maps each source model name to a list of claim strings.

    Returns a list of AtomicClaim with sequential claim_id values that are unique across all source models.
    Empty strings are skipped.
    """
    claims: List[AtomicClaim] = []
    claim_id = 0
    for source_model, texts in claims_by_model.items():
        for text in texts:
            text = text.strip()
            if not text:
                continue
            claims.append(AtomicClaim(claim_id=claim_id, text=text, source_model=source_model))
            claim_id += 1
    return claims


@dataclass
class TruthFinderResult:
    """Output of truthfinder with trust for each source, confidence for each claim, convergence data and an optional history."""
    source_trustworthiness: Dict[str, float]
    claim_confidence: Dict[int, float]   # keyed by claim id
    converged: bool
    n_iterations: int
    history: Optional[Dict[str, Any]] = None  # set only when return_history is true


# ENTAILMENT MATRIX

# entries are signed scores in [-1; 1] with positive for entailment and negative for contradiction
# zero means no signal

def build_entailment_matrix(
    claims: List[AtomicClaim],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
    sas_model_name: str = "cross-encoder/stsb-roberta-large",
    sas_threshold: float = 0.35,
    temperature: float = 1.0,
    deadzone_threshold: float = 0.10,
    incoherence_threshold: float = 0.10,
    batch_size: int = 32,
    device: Optional[str] = None,
    debug: bool = True,
    llm_judge: bool = False,
    llm_judge_weight: float = 0.5,
    combine_mode: Literal["weighted", "min", "max"] = "weighted",
    llm_client: Any = None,
    llm_model: str = "",
    llm_temperature: Optional[float] = None,
    positive_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the signed entailment matrix and the similarity matrix for a list of AtomicClaim.

    For each pair of claims from different models, the larger of the two directional similarity scores gates which pairs reach the NLI model and the LLM judge.
    The returned similarity matrix keeps the directional scores in [0; 1].
    The NLI score is P(entailment) - P(contradiction).
    An optional LLM judge scores the same pairs, and the two scores are combined with combine_mode.
    Scores below deadzone_threshold in absolute value are set to zero.
    When both directions of a pair are strong and have opposite signs, both are set to zero.
    When positive_only is true, negative values are set to zero at the end.

    Returns the entailment matrix and the similarity matrix, both with shape N by N.
    """

    def _log(msg: str):
        if debug:
            print(f"build_entailment_matrix: {msg}")

    # automatic device choice when none is given
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(resolved_device)
    # half precision only on gpu
    use_fp16 = dev.type == "cuda"

    # use the values set by configure_judge when none are passed
    if llm_judge and llm_client is None:
        llm_client = _llm_client
    if llm_judge and not llm_model:
        llm_model = LLM_MODEL
    if llm_judge and llm_temperature is None:
        llm_temperature = LLM_TEMP

    texts = [c.text for c in claims]
    n = len(claims)

    matrix = np.zeros((n, n), dtype=np.float64)
    sas_matrix = np.zeros((n, n), dtype=np.float64)

    # pairs from different models only
    pair_idx = [
        (i, j) for i in range(n) for j in range(n)
        if i != j and claims[i].source_model != claims[j].source_model
    ]
    _log(f"Claims: {n}, sources: {len(set(c.source_model for c in claims))}, "
         f"cross-model pairs: {len(pair_idx)}")

    # without cross-model pairs every score is zero
    if not pair_idx:
        _log("No cross-model pairs: matrices are zero")
        return matrix, sas_matrix

    p_total = len(pair_idx)
    all_premises = [texts[i] for i, j in pair_idx]
    all_hypotheses = [texts[j] for i, j in pair_idx]

    nli_scores_global = np.zeros(p_total, dtype=np.float32)
    # marks the pairs that received an NLI score
    nli_valid_mask = np.zeros(p_total, dtype=bool)
    llm_scores_global = np.zeros(p_total, dtype=np.float32)

    # NLI is skipped only when the judge is on and has weight one
    need_nli = llm_judge_weight != 1.0
    need_sas = True  # SAS always runs and gates NLI and the LLM judge

    # SAS SCORES
    sas_filtered_indices: List[int] = []
    if need_sas:
        _log(f"Computing SAS with {sas_model_name}")
        sas_tokenizer = AutoTokenizer.from_pretrained(sas_model_name)
        sas_model = AutoModelForSequenceClassification.from_pretrained(sas_model_name).to(dev)
        sas_model.eval()
        if use_fp16:
            sas_model = sas_model.half()

        sas_scores = np.zeros(p_total, dtype=np.float32)
        with torch.no_grad():
            for start in tqdm(range(0, p_total, batch_size), desc="SAS batches", disable=not debug,
                              total=math.ceil(p_total / batch_size)):
                end = start + batch_size
                enc = sas_tokenizer(
                    all_premises[start:end], all_hypotheses[start:end],
                    padding=True, truncation=True, max_length=512, return_tensors="pt",
                ).to(dev)
                logits = sas_model(**enc).logits.squeeze(-1)
                sas_scores[start:end] = torch.sigmoid(logits.float()).cpu().numpy()  # sigmoid for the regression head

        # free the model before the next one is loaded
        del sas_model, sas_tokenizer
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        for k, (i, j) in enumerate(pair_idx):
            sas_matrix[i, j] = float(sas_scores[k])

        # use the larger score of the two directions
        sas_pair_max: Dict[frozenset, float] = {}
        for k, (i, j) in enumerate(pair_idx):
            key = frozenset((i, j))
            sas_pair_max[key] = max(sas_pair_max.get(key, 0.0), float(sas_scores[k]))

        # a pair passes when either direction reaches the threshold
        sas_filtered_indices = [
            k for k, (i, j) in enumerate(pair_idx)
            if sas_pair_max[frozenset((i, j))] >= sas_threshold
        ]
        _log(f"SAS filter with threshold {sas_threshold}: keeps "
             f"{len(sas_filtered_indices)} of {p_total} pairs")

    # NLI SCORES
    # only pairs that passed the SAS filter are scored
    if need_nli and sas_filtered_indices:
        _log(f"Loading NLI model {nli_model_name}")
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(dev)
        nli_model.eval()
        if use_fp16:
            nli_model = nli_model.half()

        # find the index of each label from the model config
        id2label = {k: v.lower() for k, v in nli_model.config.id2label.items()}
        label2idx: Dict[str, int] = {}
        for idx, name in id2label.items():
            if "entail" in name:
                label2idx["entailment"] = idx
            elif "contra" in name:
                label2idx["contradiction"] = idx
            elif "neutral" in name:
                label2idx["neutral"] = idx
        # default order when the config names are not recognised
        if set(label2idx.keys()) != {"contradiction", "neutral", "entailment"}:
            label2idx = {"contradiction": 0, "neutral": 1, "entailment": 2}
        _log(f"Label indices: {label2idx}")

        premises_f = [all_premises[k] for k in sas_filtered_indices]
        hypotheses_f = [all_hypotheses[k] for k in sas_filtered_indices]
        p_f = len(sas_filtered_indices)

        for start in tqdm(range(0, p_f, batch_size), desc="NLI batches", disable=not debug,
                          total=math.ceil(p_f / batch_size)):
            end = start + batch_size
            enc = nli_tokenizer(
                premises_f[start:end], hypotheses_f[start:end],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                logits = nli_model(**enc).logits
            # temperature above one softens the probabilities
            probs = F.softmax(logits.float() / temperature, dim=-1)
            entail_p = probs[:, label2idx["entailment"]]
            contra_p = probs[:, label2idx["contradiction"]]
            signed = (entail_p - contra_p).cpu().numpy()  # score in [-1; 1]
            chunk_indices = sas_filtered_indices[start:end]
            nli_scores_global[chunk_indices] = signed
            nli_valid_mask[chunk_indices] = True

        # free the model
        del nli_model, nli_tokenizer
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        _log(f"NLI done on {p_f} pairs")

    # LLM JUDGE
    # the judge is switched off when it cannot run
    if llm_judge and llm_client is None:
        _log("No LLM client: judge off")
        llm_judge = False

    if llm_judge and not sas_filtered_indices:
        _log("No pairs passed SAS: judge off")
        llm_judge = False

    # score the pairs that passed SAS in batches
    if llm_judge:
        _log(f"LLM judge on {len(sas_filtered_indices)} pairs, "
             f"combine_mode: {combine_mode}, batch size: {batch_size}")
        n_judge_pairs = len(sas_filtered_indices)
        n_judge_batches = math.ceil(n_judge_pairs / batch_size)
        total_scores_received = 0

        for batch_start in tqdm(range(0, n_judge_pairs, batch_size), desc="LLM judge batches",
                                 disable=not debug, total=n_judge_batches):
            batch_end = batch_start + batch_size
            batch_global_indices = sas_filtered_indices[batch_start:batch_end]
            n_in_batch = len(batch_global_indices)

            # each pair is numbered within the batch
            pairs_text = "\n".join(
                f'{local_idx}. PREMISE: "{all_premises[k]}" | HYPOTHESIS: "{all_hypotheses[k]}"'
                for local_idx, k in enumerate(batch_global_indices)
            )

            prompt_text = (
                "You are an expert in formal semantics, entailment and contradiction.\n"
                f"There are exactly {n_in_batch} numbered pairs below, indexed 0 to {n_in_batch - 1}.\n"
                "For each pair, score the logical relation between PREMISE and HYPOTHESIS from\n"
                "-1.0 to 1.0:\n"
                "  +1.0  = PREMISE clearly entails HYPOTHESIS\n"
                "   0.0  = PREMISE and HYPOTHESIS are unrelated / neutral\n"
                "  -1.0  = PREMISE clearly contradicts HYPOTHESIS\n"
                "Use an intermediate value only when the relation is genuinely partial or\n"
                "mixed. When the logical relation is clear, use a score close to the\n"
                "corresponding endpoint. Reserve 0.0 for pairs that are genuinely unrelated,\n"
                "not for cases where you are simply unsure.\n\n"
                "PAIRS:\n" f"{pairs_text}\n\n"
                f"Return ONLY a valid JSON object with exactly {n_in_batch} entries, mapping each\n"
                "pair index (as a string) to its score.\n"
                "Example format: {\"0\": 0.0, \"1\": 1.0, \"2\": -0.5}\n"
                "No explanation, no extra text, no markdown."
            )


            try:
                raw = chat_completion_text(
                    llm_client,
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=llm_temperature,
                )
                # take the JSON object out of the reply
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(match.group(0)) if match else {}

                # warn when the judge skipped or added entries
                if len(parsed) != n_in_batch:
                    _log(f"Warning: batch {batch_start} expected {n_in_batch} scores, "
                         f"got {len(parsed)}")

                # missing scores count as zero and values are clipped to [-1; 1]
                for local_idx, k in enumerate(batch_global_indices):
                    llm_scores_global[k] = float(np.clip(parsed.get(str(local_idx), 0.0), -1.0, 1.0))
                total_scores_received += len(parsed)
            except Exception as e:
                # scores not yet written for a failed batch stay zero
                _log(f"LLM judge batch {batch_start} failed: {e}")

        _log(f"LLM judge returned {total_scores_received} of {n_judge_pairs} scores "
             f"in {n_judge_batches} batches")

    # COMBINE NLI AND LLM
    # without the judge the NLI scores are final
    if llm_judge:
        w = float(np.clip(llm_judge_weight, 0.0, 1.0))
        if not need_nli:
            # no NLI scores exist so combine_mode has no effect
            final_scores = llm_scores_global
            _log("NLI skipped: LLM judge scores are used directly")
        else:
            # weighted mean, minimum or maximum of the two scores
            if combine_mode == "weighted":
                combined = (1.0 - w) * nli_scores_global + w * llm_scores_global
            elif combine_mode == "min":
                combined = np.minimum(nli_scores_global, llm_scores_global)
            elif combine_mode == "max":
                combined = np.maximum(nli_scores_global, llm_scores_global)
            else:
                raise ValueError(f"Unknown combine_mode: {combine_mode}")
            # pairs without an NLI score use the LLM score alone
            combined = np.where(nli_valid_mask, combined, llm_scores_global)
            final_scores = combined
            _log(f"Combined NLI and LLM with mode: {combine_mode}, "
                 f"weight: {w if combine_mode == 'weighted' else 'none'}")
    else:
        final_scores = nli_scores_global

    # DEADZONE
    # small scores are treated as no signal
    n_zeroed = int((np.abs(final_scores) < deadzone_threshold).sum())
    final_scores[np.abs(final_scores) < deadzone_threshold] = 0.0
    _log(f"Deadzone: zeroed {n_zeroed} pairs at threshold {deadzone_threshold}")

    # write the scores back into the matrix
    cross_i = [i for i, j in pair_idx]
    cross_j = [j for i, j in pair_idx]
    matrix[cross_i, cross_j] = final_scores

    # INCOHERENCE FILTER
    n_incoherent = 0
    for i, j in pair_idx:
        # each unordered pair is checked once
        if i >= j:
            continue
        mij, mji = matrix[i, j], matrix[j, i]
        same_sign_strong = (mij * mji) < 0  # true when the signs differ
        # strong scores with opposite signs in the two directions cancel out
        if (abs(mij) >= incoherence_threshold and abs(mji) >= incoherence_threshold
                and same_sign_strong):
            matrix[i, j] = matrix[j, i] = 0.0
            n_incoherent += 1
    _log(f"Incoherence filter: zeroed {n_incoherent} pairs at threshold {incoherence_threshold}")

    # POSITIVE ONLY
    # contradiction edges are removed on request
    if positive_only:
        n_dropped = int((matrix < 0).sum())
        matrix[matrix < 0] = 0.0
        _log(f"positive_only removed {n_dropped} contradiction edges")

    n_pos = int((matrix > 0).sum())
    n_neg = int((matrix < 0).sum())
    _log(f"Final matrix: shape {matrix.shape}, {n_pos} positive edges, {n_neg} negative edges")

    return matrix, sas_matrix


# TRUTHFINDER

# isolated means no positive incoming edge so a contradicted claim without support is still isolated
# tau_clip bounds weight, state and updated state

def truthfinder(
    claims: List[AtomicClaim],
    implication_matrix: np.ndarray,
    initial_source_trust: Optional[Dict[str, float]] = None,
    influence_rho: float = 0.9,
    gamma: float = 1.0,
    damping: float = 0.6,
    epsilon: float = 1e-4,
    max_iterations: int = 200,
    initial_trust: float = 0.5,
    tau_clip: float = 4.0,
    use_surprise_penalty: bool = True,
    prior_weight: float = None,
    surprise_penalty_override: Optional[float] = None,
    adj_softcap: Optional[float] = 0.5,
    equivalence_pull: float = 0.7,
    debug: bool = False,
    return_history: bool = False,
) -> TruthFinderResult:
    """
    Run TruthFinder on atomic claims from several sources.

    The implication matrix holds signed scores in [-1; 1], with positive values for entailment, negative values for contradiction and zero for no signal.

    prior_weight sets how strongly trust is pulled toward the initial trust of each source.
    At each iteration the weight and the state of a claim are mixed as (1 - prior_weight) * current + prior_weight * prior, where the prior comes from the initial trust of its source.
    With zero the prior has no effect.
    When it is None, it is computed as 0.3 * (1 - N_neg / n) + 2 / n where N_neg is the number of negative entries and n the number of claims, and kept in [0.05; 0.95].

    surprise_penalty_override sets one flat penalty value, in logit units, applied only to isolated claims.
    use_surprise_penalty false disables the penalty and the override.
    When it is None, the penalty of a claim is log1p(-k * log(1 - p_bar)) where k is the number of claims from other sources.
    Here p_bar = (P + 1) / (C + 2) where P is the number of positive entries and C is the number of ordered cross-source pairs.
    The penalty applies only to isolated claims, which are claims with no positive incoming edge.
    The floor of an isolated claim is logit(current trust of its own source) - penalty.

    adj_softcap limits with a tanh the net positive adjustment of a claim, so the bonus never exceeds adj_softcap in logit units.
    This stops sources that confirm each other from pushing trust to saturation.
    The net negative part, present when contradictions outweigh support, is not limited.
    With None the bonus is not limited.

    equivalence_pull moves each claim toward its strongest mutual implication partner.
    The partner j of claim i maximizes w = max(0; M[i, j]) * max(0; M[j, i]) where M is the implication matrix.
    When the state of j is higher than the state of i, the state of i moves by equivalence_pull * w * (state_j - state_i).
    equivalence_pull is in [0; 1].
    The pull never lowers a state and never goes beyond the state of the partner.
    At the value one the update is a convex combination weighted by w.
    A lower value discounts a noisy matrix.

    Returns a TruthFinderResult with source trustworthiness, claim confidence, convergence data and an optional history.
    """

    eps_clip = 1e-6

    def _log(msg: str):
        if debug:
            print(f"truthfinder: {msg}")

    def _clip(p: np.ndarray, eps: float = eps_clip) -> np.ndarray:
        return np.clip(p, eps, 1.0 - eps)

    # the matrix must have one row and one column per claim
    n = len(claims)
    if implication_matrix.shape != (n, n):
        raise ValueError(f"Matrix shape {implication_matrix.shape} does not match {n} claims")

    models = sorted(set(c.source_model for c in claims))

    _log(f"Start with {n} claims, sources: {', '.join(models)}, "
         f"influence_rho: {influence_rho}, gamma: {gamma}, damping: {damping}, epsilon: {epsilon}, "
         f"max_iterations: {max_iterations}, initial_trust: {initial_trust}, tau_clip: {tau_clip}, "
         f"adj_softcap: {adj_softcap}")

    # initial_trust for every source unless initial_source_trust gives its own
    if initial_source_trust is None:
        t = {m: initial_trust for m in models}
        _log(f"Initial trust uniform: {t}")
    else:
        t = {m: initial_source_trust.get(m, initial_trust) for m in models}
        _log(f"Initial trust given: {t}")

    if prior_weight is None:
        # prior_weight comes from the negative entries per claim and from the number of claims
        a, b = 0.3, 2.0
        n_negative = (implication_matrix < 0).sum()
        neg_ratio = n_negative / n
        prior_weight = a * (1 - neg_ratio) + b / n
        prior_weight = float(np.clip(prior_weight, 0.05, 0.95))
        _log(f"Automatic prior_weight: {prior_weight:.4f}, neg_ratio: {neg_ratio:.4f}, n: {n}")

    t_prior = dict(t)  # initial trust used as prior in Step A
    if prior_weight > 0:
        _log(f"Prior active with prior_weight: {prior_weight}, prior: {t_prior}")

    source_of = [c.source_model for c in claims]
    claims_by_source: Dict[str, List[int]] = {m: [] for m in models}
    for idx, c in enumerate(claims):
        claims_by_source[c.source_model].append(idx)
    for m in models:
        _log(f"Source {m} has {len(claims_by_source[m])} claims")

    in_degree_pos = (implication_matrix > 0).sum(axis=0)
    in_degree_neg = (implication_matrix < 0).sum(axis=0)
    isolated_mask = (in_degree_pos == 0)  # no positive incoming edge

    if surprise_penalty_override is not None:
        # fixed penalty with the same value for every claim
        opportunities = np.array([
            sum(1 for c2 in claims if c2.source_model != c.source_model) for c in claims
        ])
        p_bar = float("nan")
        surprise_penalty_per_claim = np.full(n, float(surprise_penalty_override))
    else:
        # number of claims from other models that each claim is compared with
        opportunities = np.array([
            sum(1 for c2 in claims if c2.source_model != c.source_model) for c in claims
        ])
        n_cross_pairs_total = int(opportunities.sum())
        n_positive = (implication_matrix > 0).sum()
        p_bar = (n_positive + 1) / (n_cross_pairs_total + 2)
        # the penalty grows with opportunities and log1p slows the growth
        surprise_penalty_per_claim = -opportunities * math.log(max(1.0 - p_bar, 1e-9))
        surprise_penalty_per_claim = np.log1p(surprise_penalty_per_claim)

    _log(f"In-degree positive min {in_degree_pos.min()}, max {in_degree_pos.max()}, "
         f"negative min {in_degree_neg.min()}, max {in_degree_neg.max()}, "
         f"isolated claims: {int(isolated_mask.sum())}")
    if use_surprise_penalty:
        if surprise_penalty_override is not None:
            _log(f"Surprise penalty fixed at {surprise_penalty_override:.4f} for isolated claims")
        else:
            _log(f"Surprise penalty active with p_bar: {p_bar:.4f}, "
                 f"range over isolated claims: "
                 f"{surprise_penalty_per_claim[isolated_mask].min() if isolated_mask.any() else 0:.3f} to "
                 f"{surprise_penalty_per_claim[isolated_mask].max() if isolated_mask.any() else 0:.3f}")

    converged = False
    n_iterations = 0
    last_delta = None
    s_adj = np.full(n, initial_trust)

    # neutral anchor computed once from the initial mean trust of all sources
    mean_t_initial = float(np.mean(list(t.values())))
    mean_t_initial_clipped = min(max(mean_t_initial, eps_clip), 1.0 - eps_clip)
    state_anchor_neutral_fixed = math.log(mean_t_initial_clipped / (1.0 - mean_t_initial_clipped))

    trust_history: Dict[str, List[float]] = {m: [] for m in models}
    conf_history: List[np.ndarray] = []

    if return_history:
        # store iteration zero with each claim confidence set to the initial trust of its source
        s_adj_seed = np.array([t[source_of[i]] for i in range(n)])
        for m in models:
            trust_history[m].append(t[m])
        conf_history.append(s_adj_seed)

    # main loop with five steps A to E
    for iteration in range(1, max_iterations + 1):
        n_iterations = iteration
        _log(f"Iteration {iteration}")

        t_vec = np.array([t[source_of[i]] for i in range(n)])
        s_raw = _clip(t_vec)
        _log(f"Step A s_raw: min {s_raw.min():.4f}, max {s_raw.max():.4f}, mean {s_raw.mean():.4f}")

        # weight is never negative and sets how strongly a claim pushes others in Step B
        # the sign of the push comes only from the sign of the matrix entry
        weight = -np.log(1.0 - s_raw)
        weight = np.clip(weight, 0.0, tau_clip)

        # state is the logit of trust and the sigmoid in Step C inverts it when gamma is one
        state = np.log(s_raw / (1.0 - s_raw))
        state = np.clip(state, -tau_clip, tau_clip)

        if prior_weight > 0.0:
            # the prior is mixed into both weight and state in the same way
            t_prior_vec = np.array([t_prior[source_of[i]] for i in range(n)])
            t_prior_vec = _clip(t_prior_vec)

            weight_prior_vec = np.clip(-np.log(1.0 - t_prior_vec), 0.0, tau_clip)
            weight = (1.0 - prior_weight) * weight + prior_weight * weight_prior_vec

            state_prior_vec = np.clip(np.log(t_prior_vec / (1.0 - t_prior_vec)), -tau_clip, tau_clip)
            state = (1.0 - prior_weight) * state + prior_weight * state_prior_vec

            _log(f"Step A prior_weight {prior_weight}: "
                 f"weight min {weight.min():.4f}, max {weight.max():.4f}, "
                 f"state min {state.min():.4f}, max {state.max():.4f}")

        # each claim receives the weighted push of the claims that point to it
        raw_adjustment = implication_matrix.T @ weight
        adjustment = influence_rho * raw_adjustment

        positive_part = np.maximum(adjustment, 0.0)
        negative_part = np.minimum(adjustment, 0.0)  # net negative part, contradictions outweighing support

        positive_part_uncapped = positive_part  # kept for the log before the softcap
        if adj_softcap is not None:
            # limit the mutual reinforcement bonus to stop feedback loops between sources
            # the net negative part is left unchanged
            positive_part = adj_softcap * np.tanh(positive_part / adj_softcap)

        # the surprise penalty pushes isolated claims down but only as far as a floor
        # a real contradiction has no limit
        penalty_term = np.zeros(n)
        if use_surprise_penalty:
            # the floor is the logit of the source trust minus the penalty
            per_source_anchor = np.empty(n)
            # logit of the trust of the source of each claim
            for m_src, idxs_src in claims_by_source.items():
                t_src = min(max(t[m_src], eps_clip), 1.0 - eps_clip)
                a = math.log(t_src / (1.0 - t_src))
                for ii in idxs_src:
                    per_source_anchor[ii] = a

            # non-isolated claims are not penalised and get a neutral anchor
            state_anchor = np.full(n, state_anchor_neutral_fixed)
            state_anchor = np.where(isolated_mask, per_source_anchor, state_anchor)

            # the penalty stops once the state reaches the floor
            effective_penalty = surprise_penalty_per_claim
            surprise_target = state_anchor - effective_penalty
            gap = np.maximum(state - surprise_target, 0.0)
            penalty_term = -np.minimum(effective_penalty, gap)

            # only isolated claims are penalised
            penalty_term = np.where(isolated_mask, penalty_term, 0.0)
        # new state adds the bonus, the malus, the penalty to the old state
        state_star = state + positive_part + negative_part + penalty_term
        state_star = np.clip(state_star, -tau_clip, tau_clip)

        # claims with a mutual implication partner move toward the partner state in proportion to w
        # the states are frozen first so the order of the claims does not matter
        if equivalence_pull > 0.0:
            # mutual weight is the product of the two positive directions
            W_mut = np.maximum(implication_matrix, 0.0) * np.maximum(implication_matrix.T, 0.0)
            np.fill_diagonal(W_mut, 0.0)
            frozen_state = state_star.copy()
            for i_pull in range(n):
                w_max = W_mut[i_pull].max()
                # no mutual partner for this claim
                if w_max <= 0.0:
                    continue
                # among partners with equal weight pick the one with the highest state
                tied = np.flatnonzero(W_mut[i_pull] >= w_max - 1e-12)
                j_pull = int(tied[np.argmax(frozen_state[tied])])
                # the pull only raises a state and never passes the partner
                if frozen_state[j_pull] > frozen_state[i_pull]:
                    state_star[i_pull] = frozen_state[i_pull] + \
                        equivalence_pull * W_mut[i_pull, j_pull] * \
                        (frozen_state[j_pull] - frozen_state[i_pull])

        n_bonus = int(np.sum(positive_part > 1e-9))
        n_malus = int(np.sum((negative_part < -1e-9) | (penalty_term < -1e-9)))
        _log(f"Step B: {n_bonus} claims with bonus, {n_malus} with malus, "
             f"mean adjustment before equivalence pull {(positive_part + negative_part + penalty_term).mean():+.4f}")
        if adj_softcap is not None:
            n_capped = int(np.sum((positive_part_uncapped - positive_part) > 1e-6))
            if n_capped > 0:
                _log(f"Step B adj_softcap {adj_softcap}: compressed the bonus of {n_capped} claims, "
                     f"max from {positive_part_uncapped.max():.4f} to {positive_part.max():.4f}")

        # sigmoid turns the state back into a confidence
        s_adj = 1.0 / (1.0 + np.exp(-gamma * state_star))
        s_adj = _clip(s_adj)
        _log(f"Step C s_adj: min {s_adj.min():.4f}, max {s_adj.max():.4f}, mean {s_adj.mean():.4f}")

        t_new = {}
        for m in models:
            # trust of a source is the mean confidence of its claims
            idxs = claims_by_source[m]
            evidence = float(np.mean(s_adj[idxs])) if idxs else t[m]
            t_new[m] = evidence  # the prior is already applied in Step A
        _log(f"Step D before damping: {t_new}")

        # damping blends old and new trust and delta is the largest change
        t_damped = {m: damping * t[m] + (1.0 - damping) * t_new[m] for m in models}
        delta = max(abs(t_damped[m] - t[m]) for m in models)
        last_delta = delta
        t = t_damped
        _log(f"Step E after damping {damping}: trust {t}, delta {delta:.6g}")

        if return_history:
            for m in models:
                trust_history[m].append(t[m])
            conf_history.append(s_adj.copy())

        # stop when the largest trust change is below epsilon
        if delta < epsilon:
            converged = True
            _log(f"Converged at iteration {iteration}: delta {delta:.6g} is below epsilon {epsilon}")
            break

    # report a run that hit the iteration limit
    if not converged:
        msg = (f"truthfinder did not converge after {max_iterations} iterations, "
               f"residual delta: {last_delta:.6g}, threshold: {epsilon}")
        print(msg)

    _log(f"Done: converged {converged}, iterations {n_iterations}, final trust {t}")

    claim_confidence = {c.claim_id: float(s_adj[i]) for i, c in enumerate(claims)}
    history = {"trust": trust_history, "confidence": conf_history} if return_history else None

    return TruthFinderResult(
        source_trustworthiness=t,
        claim_confidence=claim_confidence,
        converged=converged,
        n_iterations=n_iterations,
        history=history,
    )


# UNSUPPORTED CLAIMS

def flag_unsupported_claims(
    claims: List[AtomicClaim],
    implication_matrix: np.ndarray,
    min_indegree: int = 1,
    positive_only: bool = True,
) -> List[Tuple[int, AtomicClaim]]:
    """
    Return the claims with fewer than min_indegree incoming edges, each with its index.
    When positive_only is true, only entailment edges count as support, so a contradicted claim without support is still flagged.
    """
    n = len(claims)
    if implication_matrix.shape != (n, n):
        raise ValueError(f"Matrix shape {implication_matrix.shape} does not match {n} claims")
    # support counts only positive edges or any edge
    if positive_only:
        in_degree = (implication_matrix > 0).sum(axis=0)
    else:
        in_degree = (implication_matrix != 0).sum(axis=0)
    return [(i, claims[i]) for i in range(n) if in_degree[i] < min_indegree]


# JUDGE CONFIGURATION

_llm_client = None
LLM_MODEL = ""
LLM_TEMP = 0.0


def configure_judge(base_url: str, api_key: str,
                     model: str,
                     temperature: float = 0.0) -> None:
    """Set the global OpenAI-compatible client, model and temperature used by the LLM judge."""
    from openai import OpenAI
    global _llm_client, LLM_MODEL, LLM_TEMP
    LLM_MODEL = model
    LLM_TEMP = temperature
    _llm_client = OpenAI(base_url=base_url, api_key=api_key)


# HISTORY PLOT

def plot_truthfinder_history(
    claims: List[AtomicClaim],
    history: Dict[str, Any],
    skip_iteration_zero: bool = True,
) -> None:
    """
    Plot source trustworthiness and claim confidence over iterations in two panels.

    The trust panel always starts at iteration zero.
    The confidence panel skips iteration zero when skip_iteration_zero is true and at least one iteration follows iteration zero.
    Both values stay in the history dict.
    """
    trust_history = history["trust"]
    conf_history = history["confidence"]
    n_iter = len(conf_history)  # includes iteration zero
    trust_start_idx = 0
    # the confidence panel can skip the starting state
    conf_start_idx = 1 if (skip_iteration_zero and n_iter > 1) else 0
    trust_iterations = list(range(trust_start_idx, n_iter))
    conf_iterations = list(range(conf_start_idx, n_iter))

    # wide figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # trust panel with one line per source
    models = list(trust_history.keys())
    colors_src = cm.tab10.colors
    for idx, m in enumerate(models):
        ax1.plot(trust_iterations, trust_history[m][trust_start_idx:], marker="o", markersize=3,
                color=colors_src[idx % len(colors_src)], label=m)
    ax1.set_title("Source Trustworthiness over iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Trustworthiness")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # one color per source model
    models_order = sorted(dict.fromkeys(c.source_model for c in claims))
    model_color = {m: colors_src[i % len(colors_src)] for i, m in enumerate(models_order)}

    # confidence panel with one line per claim
    # spread the badges evenly between 10 and 80 percent of the iteration range
    x_start = min(conf_iterations) + (max(conf_iterations) - min(conf_iterations)) * 0.1
    x_end = min(conf_iterations) + (max(conf_iterations) - min(conf_iterations)) * 0.8
    badge_positions_x = np.linspace(x_start, x_end, len(claims))

    for idx_c, c in enumerate(claims):
        conf_over_time = [conf_history[it][idx_c] for it in range(conf_start_idx, n_iter)]

        # solid lines
        ax2.plot(conf_iterations, conf_over_time, linestyle="-", linewidth=1.4,
                color=model_color[c.source_model], label=f"c{c.claim_id} [{c.source_model}]", alpha=0.85)

        # place each badge on its curve
        x_badge = badge_positions_x[idx_c]
        y_badge = np.interp(x_badge, conf_iterations, conf_over_time)

        # circle badge with the claim id
        ax2.text(
            x_badge, y_badge, str(c.claim_id), color='white', weight='bold',
            va='center', ha='center', size=8, zorder=5,
            bbox=dict(boxstyle='circle,pad=0.25', facecolor=model_color[c.source_model], edgecolor='white', lw=1)
        )

    ax2.set_title("Claim Confidence over iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ENTAILMENT GRAPH

def _separate_close_pairs(local_pos: Dict[int, np.ndarray], comp_nodes: List[int],
                           min_dist: float, max_iterations: int = 50) -> Dict[int, np.ndarray]:
    """Push apart only the pairs of nodes that are closer than min_dist and leave the other pairs unchanged."""
    coords = {node: np.array(local_pos[node], dtype=float) for node in comp_nodes}
    rng = np.random.default_rng(0)
    for _ in range(max_iterations):
        violated = False
        for a_idx in range(len(comp_nodes)):
            for b_idx in range(a_idx + 1, len(comp_nodes)):
                a, b = comp_nodes[a_idx], comp_nodes[b_idx]
                delta = coords[a] - coords[b]
                dist = float(np.linalg.norm(delta))
                # push the two nodes apart when they are too close
                if dist < min_dist:
                    violated = True
                    if dist < 1e-9:
                        delta = rng.normal(size=2)  # random direction when points coincide
                        dist = float(np.linalg.norm(delta))
                    direction = delta / dist
                    deficit = (min_dist - dist) / 2.0
                    coords[a] = coords[a] + direction * deficit
                    coords[b] = coords[b] - direction * deficit
        # stop when no pair is too close
        if not violated:
            break
    return coords


def _compute_entailment_graph_layout(G: "nx.DiGraph", isolated_nodes: List[int],
                                     connected_nodes: List[int], layout_seed: int) -> Dict[int, np.ndarray]:
    """Compute node positions in two parts.
    Connected components are packed row by row and isolated nodes are placed in a separate grid."""
    pos: Dict[int, np.ndarray] = {}

    if connected_nodes:
        H = G.subgraph(connected_nodes)
        components = sorted(nx.weakly_connected_components(H), key=len, reverse=True)

        # spacing scales with the edge length factor
        edge_length_factor = 2
        margin = 0.9 * edge_length_factor
        max_row_width = 6.0 * edge_length_factor
        cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0

        for comp_nodes in components:
            comp_nodes = list(comp_nodes)
            sub = H.subgraph(comp_nodes)

            # layout depends on the size of the component
            if len(comp_nodes) == 1:
                local_pos = {comp_nodes[0]: np.array([0.0, 0.0])}
            elif len(comp_nodes) == 2:
                local_pos = {comp_nodes[0]: np.array([-0.75, 0.0]), comp_nodes[1]: np.array([0.75, 0.0])}
            else:
                local_pos = nx.spring_layout(sub, seed=layout_seed, k=0.7 / np.sqrt(len(comp_nodes)), iterations=200)
                local_pos = _separate_close_pairs(local_pos, comp_nodes, min_dist=1.5)

            # stretch the local positions
            for node in comp_nodes:
                local_pos[node] *= edge_length_factor

            # bounding box of the component
            coords = np.array([local_pos[node] for node in comp_nodes])
            width = coords[:, 0].max() - coords[:, 0].min() if len(comp_nodes) > 1 else 0.0
            height = coords[:, 1].max() - coords[:, 1].min() if len(comp_nodes) > 1 else 0.0

            if cursor_x > 0.0 and cursor_x + width > max_row_width:
                # move down by the row height before starting a new row
                cursor_y -= row_height + margin
                cursor_x, row_height = 0.0, 0.0

            # move the component so its top-left corner is at the cursor
            comp_origin = np.array([
                coords[:, 0].min() if len(comp_nodes) > 1 else 0.0,
                coords[:, 1].max() if len(comp_nodes) > 1 else 0.0,
            ])
            offset = np.array([cursor_x, cursor_y]) - comp_origin
            for node in comp_nodes:
                pos[node] = local_pos[node] + offset

            cursor_x += width + margin
            row_height = max(row_height, height)

    # isolated nodes go in a square grid right of and below the components
    if isolated_nodes:
        n_iso = len(isolated_nodes)
        iso_cols = max(1, math.ceil(math.sqrt(n_iso)))
        iso_spacing = 1.1

        # the grid starts after the connected components when they exist
        if connected_nodes:
            conn_xs = [p[0] for p in pos.values()]
            conn_ys = [p[1] for p in pos.values()]
            origin_x = max(conn_xs) + 2.2
            origin_y = min(conn_ys) - 1.0
        else:
            origin_x, origin_y = 0.0, 0.0

        for k, node in enumerate(isolated_nodes):
            row, col = divmod(k, iso_cols)
            pos[node] = np.array([origin_x + col * iso_spacing, origin_y - row * iso_spacing])

    return pos


def plot_entailment_graph(
    claims: List[AtomicClaim],
    implication_matrix: np.ndarray,
    title: str = "Atomic Claim Entailment Graph",
    weight_threshold: float = 0.0,
    layout_seed: int = 42,
    figsize: tuple = (9, 8),
    node_size: int = 700,
    show_weights: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Draw a directed graph of entailment and contradiction between atomic claims.

    Nodes are colored by source model.
    Positive edges are gray and negative edges are red, and their width and opacity grow with the absolute weight relative to the other edges of the same sign.
    Nodes without positive incoming edges get a red outline, whether or not they have contradiction edges.
    Connected components are packed side by side and claims with no edge are placed in a separate grid.
    """
    n = len(claims)
    if implication_matrix.shape != (n, n):
        raise ValueError(f"Matrix shape {implication_matrix.shape} does not match {n} claims")

    colors_src = cm.tab10.colors
    models_order = sorted(dict.fromkeys(c.source_model for c in claims))
    model_color = {m: colors_src[i % len(colors_src)] for i, m in enumerate(models_order)}

    G = nx.DiGraph()
    for i, c in enumerate(claims):
        G.add_node(i, source_model=c.source_model, claim_id=c.claim_id, text=c.text)

    # only edges above the weight threshold are drawn
    pos_edges, neg_edges = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = implication_matrix[i, j]
            if abs(w) > weight_threshold:
                G.add_edge(i, j, weight=float(w))
                (pos_edges if w > 0 else neg_edges).append((i, j, float(w)))

    # claims without positive support get a red outline
    in_degree_pos = (implication_matrix > 0).sum(axis=0)
    no_support_nodes = [i for i in range(n) if in_degree_pos[i] == 0]

    # nodes without any edge are isolated
    isolated_nodes = [i for i in range(n) if G.in_degree(i) == 0 and G.out_degree(i) == 0]
    connected_nodes = [i for i in range(n) if i not in isolated_nodes]
    pos = _compute_entailment_graph_layout(G, isolated_nodes, connected_nodes, layout_seed)

    # create a figure only when no axis is given
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    # set the axis limits from all positions before drawing
    all_coords = np.array([pos[i] for i in range(n)])
    pad_x = max(0.8, (all_coords[:, 0].max() - all_coords[:, 0].min()) * 0.12)
    pad_y = max(0.8, (all_coords[:, 1].max() - all_coords[:, 1].min()) * 0.12)
    ax.set_xlim(all_coords[:, 0].min() - pad_x, all_coords[:, 0].max() + pad_x)
    ax.set_ylim(all_coords[:, 1].min() - pad_y, all_coords[:, 1].max() + pad_y)
    ax.autoscale(False)

    for m in models_order:
        node_ids = [i for i in range(n) if claims[i].source_model == m]
        nx.draw_networkx_nodes(G, pos, nodelist=node_ids, ax=ax,
                                node_color=[model_color[m]] * len(node_ids),
                                node_size=node_size, label=m)

    if no_support_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=no_support_nodes, ax=ax,
                                node_color="none", edgecolors="red",
                                linewidths=2.5, node_size=node_size)

    # edge width and opacity grow with the normalised absolute weight
    def _draw_edge_group(edges, color):
        if not edges:
            return
        weights = np.array([abs(w) for _, _, w in edges])
        w_min, w_max = weights.min(), weights.max()
        w_norm = (weights - w_min) / (w_max - w_min) if w_max > w_min else np.full_like(weights, 0.5)
        widths = 0.8 + 1.5 * np.sqrt(w_norm)
        alphas = 0.25 + 0.65 * w_norm
        for (i, j, w), width, alpha in zip(edges, widths, alphas):
            # curved edges when both directions exist
            rad = 0.15 if G.has_edge(j, i) else 0.0
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], ax=ax,
                                    width=width, alpha=alpha, edge_color=color,
                                    arrows=True, arrowsize=14,
                                    connectionstyle=f"arc3,rad={rad}", node_size=node_size)

    _draw_edge_group(pos_edges, "gray")
    _draw_edge_group(neg_edges, "red")

    labels = {i: f"c{claims[i].claim_id}" for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    if show_weights:
        all_edges = pos_edges + neg_edges
        for i, j, w in all_edges:
            # shift the label along the edge when both directions exist
            lbl_pos = 0.33 if G.has_edge(j, i) else 0.5

            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(i, j): f"{w:.2f}"}, ax=ax, font_size=7, label_pos=lbl_pos,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75}
            )

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

    # legend note for the red outline
    if no_support_nodes:
        ax.text(0.02, 0.02,
                 f"Red outline = no entailment support (in_degree_positive=0), n={len(no_support_nodes)}\n"
                 f"Red edges = contradiction",
                 transform=ax.transAxes, fontsize=7, color="red")

    # show the figure only when it was created here
    if standalone:
        plt.tight_layout()
        plt.show()


# EXAMPLE

if __name__ == "__main__":
    # atomic claims of two models, as returned by the decomposition step
    claims_by_model = {
        "model_a": ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        "model_b": ["Paris is the capital of France.", "The Eiffel Tower is in Berlin."],
    }

    claims = claims_from_decomposition(claims_by_model)

    # signed entailment matrix and similarity matrix
    matrix, sas_matrix = build_entailment_matrix(claims)

    result = truthfinder(claims, matrix)
    print(result.source_trustworthiness)
    print(result.claim_confidence)
