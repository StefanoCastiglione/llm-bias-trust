# A Framework for Bias Assessment and Truth Discovery on Large Language Models

Code of the system developed for the master's thesis in Computer Science at Politecnico di Milano.

Author: Stefano Castiglione

Advisor: Prof. Cinzia Cappiello



## Abstract

As Large Language Models (LLMs) are increasingly used in tasks where their answers provide decisive input for users’ decisions, it becomes important to raise awareness of their weaknesses. Their behavior depends heavily on the training data, which contains society’s false beliefs and prejudices, so responses can be biased, i.e., unfairly affected by who the question is about or how it is formulated. Moreover, models can produce hallucinations, i.e., false pieces of information, presented alongside the correct ones without any warning.

This thesis presents a framework that allows users to evaluate LLMs from the outside, in a setting where their internals are inaccessible and no external source can be safely treated as ground truth, by relying only on black-box access and on the models’ own answers.

The system is composed of two modules. The *Bias Measurement* module aims to quantify bias in the responses through prompt perturbation: the higher the semantic distance between the responses to variants of the same prompt, the higher the indication of bias. The *Truth Discovery* module aims to detect hallucinations by weighing the consensus across multiple models, with an adaptation of the TruthFinder algorithm, which computes source trustworthiness and claim confidence together.

The system was validated on synthetic scenarios and showed the intended behavior both in measuring bias and in separating the confidence of true claims from that of false ones. A small-scale experiment on real LLM responses then found, on average, lower trustworthiness for the models with worse bias scores, as expected.

Overall, this work proposes a practical and flexible evaluation method without accessing internal model parameters or other knowledge sources.



## Contents



The system has two modules, Bias Measurement and Truth Discovery, plus a shared utility. Each file includes a short usage example in its `__main__` block.



**Bias Measurement**

* `bias_pipeline.py`: prompt analysis, i.e., classification of the prompt and generation of its variants
* `bias_query.py`: querying of the evaluated model with the prompt and its variants
* `bias_metrics.py`: bias measures (Confirmation Bias and FSI, Demographic Bias and VI, ERI)
* `bias_aggregator.py`: optional aggregation of the bias scores into one score per model



**Truth Discovery**

* `claim_decomposition.py`: decomposition of the responses into atomic claims
* `truthfinder.py`: implication matrix and TruthFinder



**Shared**

* `llm_call_utils.py`: helper for calls to OpenAI-compatible LLM APIs





The system was developed in Python 3.13. Generative AI tools were used as support for developing the code in this repository. The design of the framework is the author's own work.

