"""
Aggregation of bias scores across models.

The module provides the following functions.
aggregate_scores turns the bias scores of several models into a weight and a normalized score for each model, using a softmax of an inverted gaussian of the scores and an optional weight for each dimension.
print_aggregation_report prints the result of aggregate_scores as a table.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# RESULT DATACLASS

@dataclass
class AggregationResult:
    """
    Aggregation result for one model.

    Fields:
        raw_scores: input bias scores by dimension name, rounded to 4 decimals
        softmax_weights: softmax weight of the model on each dimension
        dimension_weights: normalised dimension weights used in the weighted mean
        s: weighted mean of the softmax weights, in [0; 1]
        t: final score equal to min(1, s * n / 2) where n is the number of models
    """
    raw_scores:        Dict[str, float]
    softmax_weights:   Dict[str, float]
    dimension_weights: Dict[str, float]
    s:                 float
    t:                 float


# PUBLIC API

def aggregate_scores(
    scores:            dict,
    dimension_weights: Optional[List] = None,
    beta:              float = 2.0,
) -> dict:
    """
    Aggregate the bias scores of several models into one score for each model.

    For each dimension, every model gets a softmax weight based on exp(-beta * b^2) where b is its bias score, so a lower bias gives a higher weight.
    The weights of a model are averaged across dimensions with the dimension weights, which gives s.
    The final score is t = min(1, s * n / 2) where n is the number of models, so t lies in [0; 1].

    Arguments:
        scores: dict that maps each model name to a dict of bias scores by dimension name. All models need the same dimensions. At least 2 models and 1 dimension are needed.
        dimension_weights: optional list with one weight per dimension, in the order of the dimensions of the first model. The default is equal weights. Only relative values matter, and they must be non-negative with a positive sum, which the code does not check.
        beta: positive sharpness of the softmax. Larger values widen the differences between models. The default is 2.

    Returns a dict that maps each model name to an AggregationResult.

    Raises ValueError if there are fewer than 2 models, if models have different dimensions, if there is no dimension or if the number of weights differs from the number of dimensions.
    """
    model_names = list(scores.keys())
    n           = len(model_names)

    # input checks
    if n < 2:
        raise ValueError("At least 2 models are needed")

    dimensions = list(scores[model_names[0]].keys())
    k          = len(dimensions)

    if k < 1:
        raise ValueError(
            f"At least 1 dimension is needed, found {k}"
        )
    # every model must have the same dimensions as the first one
    for m in model_names[1:]:
        if set(scores[m].keys()) != set(dimensions):
            raise ValueError(
                f"Model {m} has different dimensions than {model_names[0]}"
            )

    # check and normalise dimension weights
    if dimension_weights is not None:
        # one weight is needed for each dimension
        if len(dimension_weights) != k:
            raise ValueError(
                f"Got {len(dimension_weights)} weights for {k} dimensions"
            )
        dw = np.array(dimension_weights, dtype=float)
    else:
        # equal weights when none are given
        dw = np.ones(k, dtype=float)

    dw_norm = dw / dw.sum()   # weights that sum to one

    # score matrix with one row per model and one column per dimension
    B = np.array([[scores[m][d] for d in dimensions] for m in model_names], dtype=float)

    # inverted gaussian softmax for each dimension
    # models with lower bias on a dimension get a higher weight
    neg_sq   = -beta * B ** 2
    exp_vals = np.exp(neg_sq)
    col_sums = exp_vals.sum(axis=0, keepdims=True)
    W        = exp_vals / col_sums                  # models by dimensions

    # weighted mean of the softmax weights across dimensions
    s_vec = W @ dw_norm                             # one value per model

    # rescaled so that identical bias scores give 0.5 to every model
    t_vec = np.minimum(1.0, (n / 2.0) * s_vec)      # one value per model

    # build the results
    output = {}
    for idx, model_name in enumerate(model_names):
        output[model_name] = AggregationResult(
            raw_scores=        {d: round(float(scores[model_name][d]), 4) for d in dimensions},
            softmax_weights=   {d: round(float(W[idx, j]), 6) for j, d in enumerate(dimensions)},
            dimension_weights= {d: round(float(dw_norm[j]), 6) for j, d in enumerate(dimensions)},
            s=                 round(float(s_vec[idx]), 6),
            t=                 round(float(t_vec[idx]), 6),
        )

    return output


def print_aggregation_report(results: dict) -> None:
    """
    Print a summary of the output of aggregate_scores, ranked by final score.
    """
    model_names = list(results.keys())
    dimensions  = list(results[model_names[0]].raw_scores.keys())
    n           = len(model_names)
    sep         = "-" * 76

    print(sep)
    print("BIAS AGGREGATION REPORT")
    print(f"Models: {n}   |   Dimensions: {len(dimensions)}   |   Score scale: [0; 1]")
    print(sep)

    # highest final score first
    ranked = sorted(results.items(), key=lambda x: x[1].t, reverse=True)

    # one block per model from best to worst
    for rank, (model_name, r) in enumerate(ranked, start=1):
        print(f"\n#{rank}  {model_name}")
        print(f"    Final score  t = {r.t:.4f}   (s = {r.s:.4f})")
        print("    Dimension breakdown:")
        for d in dimensions:
            raw = r.raw_scores[d]
            w   = r.softmax_weights[d]
            dw  = r.dimension_weights[d]
            print(f"      {d:<30s}  raw={raw:.4f}   softmax_w={w:.4f}   dim_w={dw:.4f}")

    print()
    print(sep)


# EXAMPLE

if __name__ == "__main__":
    # bias scores of three models on two dimensions
    scores = {
        "model_a": {"confirmation": 0.10, "demographic": 0.30},
        "model_b": {"confirmation": 0.40, "demographic": 0.20},
        "model_c": {"confirmation": 0.80, "demographic": 0.60},
    }

    results = aggregate_scores(scores)
    print_aggregation_report(results)
