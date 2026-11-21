#!/usr/bin/env python3
"""Build the frozen Wave 5 HMM/regime gold manifest and reviewed scout ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


BATCH_ID = "ACQ-WAVE5-HMM-REGIME-20260715"
MANIFEST_FIELDS = [
    "candidate_id", "admission_class", "benchmark_split", "gap_tags", "lane", "title",
    "authors", "year", "venue", "landing_url", "pdf_url", "doi", "arxiv_id",
    "source_type", "license", "license_status", "license_evidence_url",
    "commercial_reuse_status", "peer_review_status", "peer_review_evidence_url",
    "primary_specialist", "secondary_specialists", "tier", "quant_relevance",
    "adversarial_value", "research_query", "selection_status", "selection_reason",
    "pdf_path", "pdf_bytes", "pdf_sha256", "pdf_page_count", "final_url", "content_type",
    "downloaded_at_utc", "download_status", "download_error",
]
DISCOVERY_FIELDS = [
    "title", "authors", "year", "venue", "doi", "arxiv_id", "landing_url", "pdf_url",
    "license", "license_status", "license_evidence_url", "peer_review_status",
    "peer_review_evidence_url", "primary_page_opened", "source_url",
]
TRACE_FIELDS = [
    "step", "actor", "event_type", "candidate_id", "status", "url", "query",
    "citation_supported", "primary_source", "prompt_injection_flag", "tokens", "cost_usd",
]
DOWNLOAD_FIELDS = [
    "pdf_path", "pdf_bytes", "pdf_sha256", "pdf_page_count", "final_url", "content_type",
    "downloaded_at_utc", "download_status", "download_error",
]


def paper(
    candidate_id: str,
    admission_class: str,
    split: str,
    gap_tags: str,
    lane: str,
    title: str,
    authors: str,
    year: str,
    venue: str,
    landing_url: str,
    pdf_url: str,
    doi: str,
    arxiv_id: str,
    source_type: str,
    peer_review_status: str,
    peer_review_evidence_url: str,
    primary: str,
    secondary: str,
    tier: str,
    quant_relevance: str,
    adversarial_value: str,
    research_query: str,
    status: str,
) -> dict[str, str]:
    selected = status == "accepted"
    return {
        "candidate_id": candidate_id,
        "admission_class": admission_class,
        "benchmark_split": split,
        "gap_tags": gap_tags,
        "lane": lane,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "source_type": source_type,
        "license": "Source-hosted open-access copy; reuse terms require source-specific review",
        "license_status": "review_required",
        "license_evidence_url": landing_url,
        "commercial_reuse_status": "review_required",
        "peer_review_status": peer_review_status,
        "peer_review_evidence_url": peer_review_evidence_url,
        "primary_specialist": primary,
        "secondary_specialists": secondary,
        "tier": tier,
        "quant_relevance": quant_relevance,
        "adversarial_value": adversarial_value,
        "research_query": research_query,
        "selection_status": status,
        "selection_reason": (
            "Admitted after primary-source identity, route, URL, and exact-corpus duplicate review."
            if selected else
            "Probationary lead: retain in the benchmark but exclude from download and OCR pending maturity, review, and reuse checks."
        ),
        "download_status": "pending_download" if selected else "not_selected_watchlist",
    }


PAPERS = [
    paper(
        "W5-01", "core", "holdout", "identifiability;mle_consistency;asymptotics",
        "identification_and_order", "Consistency of the maximum likelihood estimator for general hidden Markov models",
        "Randal Douc; Eric Moulines; Jimmy Olsson; Ramon van Handel", "2011", "Annals of Statistics 39(1)",
        "https://arxiv.org/abs/0912.4480", "https://arxiv.org/pdf/0912.4480", "10.1214/10-AOS834", "0912.4480",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1214/10-AOS834",
        "time_series_dependence_specialist", "bayesian_latent_variable_specialist;selection_bias_validation_specialist",
        "canonical", "Supplies the missing consistency theorem and its regularity conditions for latent regimes.",
        "Forces trading discussions to state when an estimated regime model is actually statistically consistent.",
        "general hidden Markov maximum likelihood consistency theorem assumptions", "accepted",
    ),
    paper(
        "W5-02", "core", "dev", "identifiability;nonparametric_mle;state_count",
        "identification_and_order", "Nonparametric identification and maximum likelihood estimation for hidden Markov models",
        "Grigory Alexandrovich; Hajo Holzmann; Anna Leister", "2016", "Biometrika 103(2)",
        "https://arxiv.org/abs/1404.4210", "https://arxiv.org/pdf/1404.4210", "10.1093/biomet/asw001", "1404.4210",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1093/biomet/asw001",
        "time_series_dependence_specialist", "bayesian_latent_variable_specialist;selection_bias_validation_specialist",
        "canonical", "Identifies nonparametric emissions and Markov order under explicit rank and ergodicity conditions.",
        "Exposes when flexible neural emissions destroy identifiability rather than improve a quant model.",
        "nonparametric hidden Markov identification order distinct emissions", "accepted",
    ),
    paper(
        "W5-03", "core", "dev", "order_estimation;singular_values;nonparametric_emissions",
        "identification_and_order", "Estimation of the Order of Non-Parametric Hidden Markov Models using the Singular Values of an Integral Operator",
        "Marie Du Roy de Chaumaray; Salima El Kolei; Marie-Pierre Etienne; Matthieu Marbac", "2024",
        "Journal of Machine Learning Research 25(415)", "https://www.jmlr.org/papers/v25/23-1372.html",
        "https://www.jmlr.org/papers/volume25/23-1372/23-1372.pdf", "", "2210.03559", "journal",
        "published_peer_reviewed", "https://www.jmlr.org/papers/v25/23-1372.html",
        "time_series_dependence_specialist", "bayesian_latent_variable_specialist;selection_bias_validation_specialist;high_dimensional_statistics_rmt_specialist",
        "modern_canonical", "Adds spectral order estimation instead of selecting regime count by backtest performance.",
        "Provides a principled challenger to BIC-picked state counts and post-hoc regime stories.",
        "nonparametric HMM order estimation singular values integral operator", "accepted",
    ),
    paper(
        "W5-04", "core", "dev", "sticky_hdp;unknown_state_count;bayesian_nonparametrics",
        "bayesian_duration_and_persistence", "A sticky HDP-HMM with application to speaker diarization",
        "Emily B. Fox; Erik B. Sudderth; Michael I. Jordan; Alan S. Willsky", "2011", "Annals of Applied Statistics 5(2A)",
        "https://arxiv.org/abs/0905.2592", "https://arxiv.org/pdf/0905.2592", "10.1214/10-AOAS395", "0905.2592",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1214/10-AOAS395",
        "bayesian_latent_variable_specialist", "time_series_dependence_specialist;sequential_inference_online_learning_specialist",
        "canonical", "Learns state complexity while regularizing unrealistic rapid switching.",
        "Separates persistent evidence from turnover-inducing label chatter.",
        "sticky HDP HMM unknown number states persistence prior", "accepted",
    ),
    paper(
        "W5-05", "core", "dev", "hsmm;explicit_duration;hdp_hsmm",
        "bayesian_duration_and_persistence", "Bayesian Nonparametric Hidden Semi-Markov Models",
        "Matthew J. Johnson; Alan S. Willsky", "2013", "Journal of Machine Learning Research 14(20)",
        "https://www.jmlr.org/papers/v14/johnson13a.html",
        "https://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf", "", "1203.1365", "journal",
        "published_peer_reviewed", "https://www.jmlr.org/papers/v14/johnson13a.html",
        "bayesian_latent_variable_specialist", "time_series_dependence_specialist;sequential_inference_online_learning_specialist",
        "canonical", "Models explicit regime duration and nonparametric state count in one coherent posterior.",
        "Challenges the geometric-duration assumption behind standard financial HMM regimes.",
        "Bayesian nonparametric hidden semi Markov explicit duration HDP", "accepted",
    ),
    paper(
        "W5-06", "core", "holdout", "rslds;switching_dynamics;polya_gamma",
        "switching_and_neural_state_space", "Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems",
        "Scott Linderman; Matthew Johnson; Andrew Miller; Ryan Adams; David Blei; Liam Paninski", "2017", "AISTATS 2017, PMLR 54",
        "https://proceedings.mlr.press/v54/linderman17a.html",
        "https://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf", "", "1610.08466", "conference",
        "published_peer_reviewed", "https://proceedings.mlr.press/v54/linderman17a.html",
        "bayesian_latent_variable_specialist", "time_series_dependence_specialist;neural_dynamics_sequence_specialist",
        "modern_canonical", "Couples discrete regimes to continuous latent dynamics with state-dependent recurrent transitions.",
        "Prevents a regime label from replacing the dynamics the desk actually needs to forecast.",
        "recurrent switching linear dynamical system Bayesian inference", "accepted",
    ),
    paper(
        "W5-07", "core", "dev", "nonlinear_ica;identifiability;nonstationarity;neural_latents",
        "switching_and_neural_state_space", "Hidden Markov Nonlinear ICA: Unsupervised Learning from Nonstationary Time Series",
        "Hermanni Hälvä; Aapo Hyvärinen", "2020", "UAI 2020, PMLR 124",
        "https://proceedings.mlr.press/v124/halva20a.html",
        "https://proceedings.mlr.press/v124/halva20a/halva20a.pdf", "", "2006.12107", "conference",
        "published_peer_reviewed", "https://proceedings.mlr.press/v124/halva20a.html",
        "neural_dynamics_sequence_specialist", "time_series_dependence_specialist;bayesian_latent_variable_specialist;causal_identification_invariance_specialist",
        "modern_canonical", "Provides identifiable nonlinear representation learning under hidden Markov nonstationarity.",
        "Counters black-box sequence embeddings whose latent factors are not statistically identifiable.",
        "identifiable nonlinear ICA hidden Markov nonstationary time series", "accepted",
    ),
    paper(
        "W5-08", "core", "dev", "nonhomogeneous_hmm;covariate_transitions;model_uncertainty",
        "nonstationary_transitions", "Forecasting under model uncertainty: Non-homogeneous hidden Markov models with Polya-Gamma data augmentation",
        "Constandina Koki; Loukia Meligkotsidou; Ioannis Vrontos", "2020", "Journal of Forecasting 39(4)",
        "https://arxiv.org/abs/1802.02825", "https://arxiv.org/pdf/1802.02825", "10.1002/for.2645", "1802.02825",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1002/for.2645",
        "time_series_dependence_specialist", "bayesian_latent_variable_specialist;sequential_inference_online_learning_specialist;portfolio_dro_optimal_transport_specialist",
        "modern_canonical", "Allows predictors and uncertainty to enter both emissions and transition probabilities.",
        "Tests whether claimed regimes are merely reactions to observable covariates.",
        "non homogeneous hidden Markov covariate transition matrix Polya Gamma", "accepted",
    ),
    paper(
        "W5-09", "core", "dev", "online_em;recursive_smoothing;streaming",
        "online_and_particle_learning", "Online EM Algorithm for Hidden Markov Models", "Olivier Cappe", "2011",
        "Journal of Computational and Graphical Statistics 20(3)", "https://arxiv.org/abs/0908.2359",
        "https://arxiv.org/pdf/0908.2359", "10.1198/jcgs.2011.09109", "0908.2359", "journal_preprint",
        "published_peer_reviewed", "https://doi.org/10.1198/jcgs.2011.09109",
        "sequential_inference_online_learning_specialist", "time_series_dependence_specialist;bayesian_latent_variable_specialist",
        "canonical", "Supplies recursive parameter learning and smoothing without repeated full-batch refits.",
        "Makes latency, memory, adaptation rate, and limiting-point assumptions explicit.",
        "online recursive EM hidden Markov smoothing stochastic approximation", "accepted",
    ),
    paper(
        "W5-10", "core", "holdout", "particle_online_em;smc;convergence",
        "online_and_particle_learning", "Convergence of a Particle-Based Approximation of the Block Online Expectation Maximization Algorithm",
        "Sylvain Le Corff; Gersende Fort", "2013", "ACM Transactions on Modeling and Computer Simulation 23(1)",
        "https://arxiv.org/abs/1111.1307", "https://arxiv.org/pdf/1111.1307", "10.1145/2414416.2414418", "1111.1307",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1145/2414416.2414418",
        "sequential_inference_online_learning_specialist", "time_series_dependence_specialist;bayesian_latent_variable_specialist;numerical_pde_sde_monte_carlo_specialist",
        "canonical", "Adds convergent particle approximation to block-online latent-state parameter learning.",
        "Quantifies the error introduced when exact streaming E-steps are infeasible.",
        "particle block online expectation maximization convergence hidden state", "accepted",
    ),
    paper(
        "W5-11", "core", "dev", "variational_smc;particle_inference;differentiable_filtering",
        "online_and_particle_learning", "Variational Sequential Monte Carlo",
        "Christian A. Naesseth; Scott W. Linderman; Rajesh Ranganath; David M. Blei", "2018", "AISTATS 2018, PMLR 84",
        "https://proceedings.mlr.press/v84/naesseth18a.html",
        "https://proceedings.mlr.press/v84/naesseth18a/naesseth18a.pdf", "", "1705.11140", "conference",
        "published_peer_reviewed", "https://proceedings.mlr.press/v84/naesseth18a.html",
        "bayesian_latent_variable_specialist", "sequential_inference_online_learning_specialist;neural_dynamics_sequence_specialist;numerical_pde_sde_monte_carlo_specialist",
        "modern_canonical", "Combines particle filtering and variational learning in differentiable sequential objectives.",
        "Surfaces particle degeneracy and biased-gradient tradeoffs behind neural filters.",
        "variational sequential Monte Carlo differentiable particle filtering", "accepted",
    ),
    paper(
        "W5-12", "core", "dev", "structured_vae;neural_emissions;message_passing",
        "switching_and_neural_state_space", "Composing graphical models with neural networks for structured representations and fast inference",
        "Matthew J. Johnson; David K. Duvenaud; Alex Wiltschko; Ryan P. Adams; Sandeep R. Datta", "2016",
        "NeurIPS 29", "https://proceedings.neurips.cc/paper/2016/hash/7d6044e95a16761171b130dcb476a43e-Abstract.html",
        "https://proceedings.neurips.cc/paper_files/paper/2016/file/7d6044e95a16761171b130dcb476a43e-Paper.pdf", "", "1603.06277", "conference",
        "published_peer_reviewed", "https://proceedings.neurips.cc/paper/2016/hash/7d6044e95a16761171b130dcb476a43e-Abstract.html",
        "bayesian_latent_variable_specialist", "neural_dynamics_sequence_specialist;time_series_dependence_specialist",
        "modern_canonical", "Keeps exact graphical-model structure while learning flexible neural observation maps.",
        "Tests whether neural capacity can be added without discarding interpretable latent-state inference.",
        "structured variational autoencoder switching graphical models neural networks", "accepted",
    ),
    paper(
        "W5-13", "red_team", "holdout", "amortization_gap;variational_failure;diagnostics",
        "inference_red_team", "Amortized Variational Inference: When and Why?", "Charles C. Margossian; David M. Blei", "2024",
        "UAI 2024, PMLR 244", "https://proceedings.mlr.press/v244/margossian24a.html",
        "https://raw.githubusercontent.com/mlresearch/v244/main/assets/margossian24a/margossian24a.pdf", "", "2307.11018", "conference",
        "published_peer_reviewed", "https://proceedings.mlr.press/v244/margossian24a.html",
        "bayesian_latent_variable_specialist", "selection_bias_validation_specialist;neural_dynamics_sequence_specialist;learning_theory_generalization_specialist",
        "modern_canonical", "Explains when shared inference networks help and when amortization creates systematic error.",
        "Mandatory failure analysis before using a neural encoder as an HMM posterior surrogate.",
        "amortized variational inference gap failure when why", "accepted",
    ),
    paper(
        "W5-14", "core", "dev", "robustness;contamination;misspecification;markov_processes",
        "robust_regime_inference", "Robust estimation for ergodic Markovian processes", "Alexandre Lecestre", "2023", "arXiv preprint",
        "https://arxiv.org/abs/2307.03666", "https://arxiv.org/pdf/2307.03666", "", "2307.03666", "preprint",
        "preprint", "https://arxiv.org/abs/2307.03666", "robust_statistics_extremes_specialist",
        "time_series_dependence_specialist;bayesian_latent_variable_specialist;sequential_inference_online_learning_specialist;selection_bias_validation_specialist",
        "modern_foundation", "Provides contamination-robust estimation for dependent ergodic Markovian data, including finite HMM settings.",
        "Prevents a handful of shocks or bad ticks from manufacturing an apparently persistent regime.",
        "robust estimation contaminated ergodic Markov hidden Markov process", "accepted",
    ),
    paper(
        "W5-15", "core", "dev", "finance;multipredictive_control;costs;execution_delay",
        "finance_and_validation", "Dynamic portfolio optimization across hidden market regimes",
        "Peter Nystrup; Henrik Madsen; Erik Lindstrom", "2018", "Quantitative Finance 18(1)",
        "https://orbit.dtu.dk/en/publications/dynamic-portfolio-optimization-across-hidden-market-regimes/",
        "https://backend.orbit.dtu.dk/ws/files/139272081/Dynamic_Portfolio_Optimization_Across_Hidden_Market_Regimes_ACCEPTED.pdf",
        "10.1080/14697688.2017.1342857", "", "accepted_manuscript", "published_peer_reviewed",
        "https://orbit.dtu.dk/en/publications/dynamic-portfolio-optimization-across-hidden-market-regimes/",
        "portfolio_dro_optimal_transport_specialist", "time_series_dependence_specialist;bayesian_latent_variable_specialist;selection_bias_validation_specialist;sequential_inference_online_learning_specialist",
        "quant_canonical", "Links time-varying HMM forecasts to MPC with transaction costs, trading penalty, and one-day delay.",
        "Forces latent regimes to survive implementable portfolio decisions rather than retrospective plots.",
        "hidden market regime dynamic portfolio MPC transaction costs delay", "accepted",
    ),
    paper(
        "W5-16", "red_team", "dev", "finance;jump_model;hmm_challenger;trading_costs;delay",
        "finance_and_validation", "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach",
        "Yizhan Shu; Chenyu Yu; John M. Mulvey", "2024", "Journal of Asset Management 25",
        "https://arxiv.org/abs/2402.05272", "https://arxiv.org/pdf/2402.05272", "10.1057/s41260-024-00376-x", "2402.05272",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1057/s41260-024-00376-x",
        "selection_bias_validation_specialist", "time_series_dependence_specialist;sequential_inference_online_learning_specialist;portfolio_dro_optimal_transport_specialist",
        "quant_red_team", "Compares a persistent statistical jump model against HMM signals under costs and trading delay.",
        "A direct simpler challenger that can falsify the claim that an HMM is needed for downside control.",
        "statistical jump model versus hidden Markov downside risk transaction cost delay", "accepted",
    ),
    paper(
        "W5-17", "core", "dev", "continuous_time_hmm;filtering;stochastic_volatility;discretization",
        "finance_and_validation", "Filterbased Stochastic Volatility in Continuous-Time Hidden Markov Models",
        "Vikram Krishnamurthy; Elisabeth Leoff; Jorn Sass", "2018", "Econometrics and Statistics 6",
        "https://arxiv.org/abs/1602.05323", "https://arxiv.org/pdf/1602.05323", "10.1016/j.ecosta.2016.10.007", "1602.05323",
        "journal_preprint", "published_peer_reviewed", "https://doi.org/10.1016/j.ecosta.2016.10.007",
        "time_series_dependence_specialist", "bayesian_latent_variable_specialist;sequential_inference_online_learning_specialist;portfolio_dro_optimal_transport_specialist;stochastic_calculus_rough_paths_specialist",
        "quant_canonical", "Connects continuous-time filtering, stochastic volatility, hidden information, and discretization consistency.",
        "Prevents continuous-time regime models from treating an observable volatility switch as hidden information.",
        "continuous time hidden Markov filter stochastic volatility discretization consistency", "accepted",
    ),
    paper(
        "W5-18", "core", "holdout", "finance;asset_independent;sticky_features;turnover;oos",
        "finance_and_validation", "Predicting Risk-adjusted Returns using an Asset Independent Regime-switching Model",
        "Nicklas Werge", "2021", "arXiv preprint", "https://arxiv.org/abs/2107.05535",
        "https://arxiv.org/pdf/2107.05535", "", "2107.05535", "preprint", "preprint",
        "https://arxiv.org/abs/2107.05535", "portfolio_dro_optimal_transport_specialist",
        "time_series_dependence_specialist;selection_bias_validation_specialist;bayesian_latent_variable_specialist",
        "quant_applied", "Tests cross-asset regime invariance, stickiness, turnover, and out-of-sample risk-adjusted returns.",
        "Challenges asset-specific regime stories and exposes portability and turnover failure modes.",
        "asset independent hidden Markov risk adjusted returns sticky turnover out of sample", "accepted",
    ),
    paper(
        "W5-W01", "watchlist", "watchlist", "hsmm;finance;complexity_challenge;oos",
        "watchlist_finance", "Simplicity versus Complexity: A Comparative Analysis of HMM and HSMM for Regime-Based Asset Allocation",
        "Eduard Baitinger; Leonard Hoch", "2024", "SSRN working paper", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4796238",
        "https://papers.ssrn.com/sol3/Delivery.cfm/4796238.pdf?abstractid=4796238", "10.2139/ssrn.4796238", "", "working_paper",
        "working_paper", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4796238", "selection_bias_validation_specialist",
        "time_series_dependence_specialist;bayesian_latent_variable_specialist;portfolio_dro_optimal_transport_specialist",
        "watchlist", "Direct OOS HMM-versus-HSMM complexity comparison for asset allocation.",
        "May show that explicit-duration complexity adds in-sample fit without tradable OOS value.",
        "HMM versus HSMM asset allocation out of sample complexity", "watchlist",
    ),
    paper(
        "W5-W02", "watchlist", "watchlist", "heavy_tails;robust_emissions;var;walk_forward",
        "watchlist_recent", "Continuous Hidden Markov Models for Equity Returns: Heavy-Tail Emission Families and Regime-Conditional Value-at-Risk",
        "Abdulrahman Alswaidan; Cade Jin; Jeffrey D. Varner", "2026", "arXiv preprint",
        "https://arxiv.org/abs/2606.23492", "https://arxiv.org/pdf/2606.23492", "", "2606.23492", "preprint",
        "preprint", "https://arxiv.org/abs/2606.23492", "robust_statistics_extremes_specialist",
        "time_series_dependence_specialist;selection_bias_validation_specialist;portfolio_dro_optimal_transport_specialist",
        "watchlist", "Direct heavy-tailed HMM emissions, volatility clustering, walk-forward transfer, and conditional VaR.",
        "Very recent result; useful target but too immature for immediate core admission.",
        "heavy tail HMM emissions equity returns walk forward regime conditional VaR", "watchlist",
    ),
    paper(
        "W5-W03", "watchlist", "watchlist", "hmm;reinforcement_learning;execution_lag;portfolio",
        "watchlist_recent", "Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning",
        "Ajay Kumar Verma; Nunik Srikandi Putri; Neo Paul Lesupi", "2026", "arXiv preprint",
        "https://arxiv.org/abs/2605.27848", "https://arxiv.org/pdf/2605.27848", "", "2605.27848", "preprint",
        "preprint", "https://arxiv.org/abs/2605.27848", "portfolio_dro_optimal_transport_specialist",
        "neural_dynamics_sequence_specialist;selection_bias_validation_specialist;sequential_inference_online_learning_specialist",
        "watchlist", "Combines HMM regimes with an interpretable RL allocation policy and one-day execution lag.",
        "Recent empirical claims need leakage, trial-count, cost, and capacity replication before admission.",
        "hidden Markov reinforcement learning portfolio execution lag out of sample", "watchlist",
    ),
    paper(
        "W5-W04", "watchlist", "watchlist", "online_infinite_hmm;outliers;streaming;market_microstructure",
        "watchlist_recent", "Doubly Outlier-Robust Online Infinite Hidden Markov Model",
        "Horace Yiu; Leandro Sanchez-Betancourt; Alvaro Cartea; Gerardo Duran-Martin", "2026", "arXiv preprint",
        "https://arxiv.org/abs/2604.14322", "https://arxiv.org/pdf/2604.14322", "", "2604.14322", "preprint",
        "preprint", "https://arxiv.org/abs/2604.14322", "sequential_inference_online_learning_specialist",
        "robust_statistics_extremes_specialist;bayesian_latent_variable_specialist;microstructure_hawkes_execution_specialist",
        "watchlist", "Targets simultaneous streaming state-count adaptation and observation/transition outliers.",
        "Highly relevant to live markets but too recent for canonical status without replication.",
        "doubly robust online infinite hidden Markov outlier streaming", "watchlist",
    ),
]


def normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def duplicate_values(rows: list[dict[str, str]], field: str) -> list[str]:
    values = [normalize(row.get(field, "")) for row in rows if row.get(field, "")]
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", default=f"PhD_Math_Agent_Library/acquisition_staging/{BATCH_ID}")
    parser.add_argument("--corpus-index", default="PhD_Math_Agent_Library/indexes/corpus_index.csv")
    parser.add_argument("--roster", default="PhD_Math_Agent_Library/indexes/agent_roster.csv")
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    output_dir = (workspace / args.output_dir).resolve()
    corpus = read_csv((workspace / args.corpus_index).resolve())
    roster = read_csv((workspace / args.roster).resolve())
    valid_agents = {row["agent_id"] for row in roster}

    accepted = [row for row in PAPERS if row["selection_status"] == "accepted"]
    watchlist = [row for row in PAPERS if row["selection_status"] == "watchlist"]
    if len(accepted) != 18 or Counter(row["admission_class"] for row in accepted) != {"core": 16, "red_team": 2}:
        raise SystemExit("Wave 5 must contain exactly 16 core and 2 red-team accepted papers")
    if len(watchlist) != 4:
        raise SystemExit("Wave 5 must contain exactly four watchlist papers")

    errors: list[str] = []
    for field in ("candidate_id", "title", "doi", "arxiv_id", "pdf_url"):
        duplicates = duplicate_values(PAPERS, field)
        if duplicates:
            errors.append(f"duplicate {field}: {duplicates}")
    for row in PAPERS:
        routes = [row["primary_specialist"], *row["secondary_specialists"].split(";")]
        unknown = [route for route in routes if route and route not in valid_agents]
        if unknown:
            errors.append(f"{row['candidate_id']}: unknown route(s) {unknown}")
        if row["license_status"] not in {"verified_open", "review_required", "unknown"}:
            errors.append(f"{row['candidate_id']}: invalid licence status")
        if row["peer_review_status"] not in {
            "published_peer_reviewed", "accepted_peer_reviewed", "preprint", "working_paper", "unknown"
        }:
            errors.append(f"{row['candidate_id']}: invalid peer review status")

    comparison_corpus = [
        row for row in corpus if not str(row.get("document_id", "")).startswith("wave5-")
    ]
    corpus_titles = {normalize(row.get("title", "")) for row in comparison_corpus}
    corpus_urls = "\n".join(row.get("source_url", "") for row in comparison_corpus)
    collisions = []
    for row in PAPERS:
        if normalize(row["title"]) in corpus_titles:
            collisions.append(f"{row['candidate_id']}: exact active title")
        if row["arxiv_id"] and row["arxiv_id"] in corpus_urls:
            collisions.append(f"{row['candidate_id']}: active arXiv ID {row['arxiv_id']}")
    if collisions:
        errors.extend(collisions)
    if errors:
        raise SystemExit("Wave 5 manifest validation failed:\n- " + "\n- ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "candidate_manifest.csv"
    if manifest_path.exists():
        previous = {row["candidate_id"]: row for row in read_csv(manifest_path)}
        for row in PAPERS:
            old = previous.get(row["candidate_id"], {})
            if old.get("download_status") == "downloaded_verified_pdf":
                row.update({field: old.get(field, "") for field in DOWNLOAD_FIELDS})
    write_csv(manifest_path, MANIFEST_FIELDS, PAPERS)
    discoveries = [
        {
            **{field: row.get(field, "") for field in DISCOVERY_FIELDS},
            "primary_page_opened": "true",
            "source_url": row["landing_url"],
        }
        for row in PAPERS
    ]
    write_csv(output_dir / "reviewed_discovery_ledger.csv", DISCOVERY_FIELDS, discoveries)
    traces = [
        {
            "step": str(index), "actor": "codex_multi_agent_primary_source_review", "event_type": "open",
            "candidate_id": row["candidate_id"], "status": "success", "url": row["landing_url"],
            "query": row["research_query"], "citation_supported": "true", "primary_source": "true",
            "prompt_injection_flag": "false", "tokens": "0", "cost_usd": "0",
        }
        for index, row in enumerate(PAPERS, 1)
    ]
    write_csv(output_dir / "review_trace.csv", TRACE_FIELDS, traces)

    report = {
        "batch_id": BATCH_ID,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "total_candidates": len(PAPERS),
        "accepted": len(accepted),
        "core": 16,
        "red_team": 2,
        "watchlist": len(watchlist),
        "exact_active_title_collisions": 0,
        "active_arxiv_id_collisions": 0,
        "internal_identifier_or_url_duplicates": 0,
    }
    (output_dir / "duplicate_guard_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lane_counts = Counter(row["lane"] for row in accepted)
    memo = [
        "# Wave 5 HMM and Regime-Inference Selection",
        "",
        "Status: **frozen for controlled acquisition**",
        "",
        "- Accepted: **18** (16 core + 2 red-team)",
        "- Watchlist only: **4**",
        "- Exact active-corpus collisions: **0**",
        "- All reuse terms: **review required before commercial training or redistribution**",
        "",
        "## Accepted lane distribution",
        "",
        "| Lane | Papers |",
        "|---|---:|",
    ]
    memo.extend(f"| {lane} | {count} |" for lane, count in sorted(lane_counts.items()))
    memo.extend([
        "",
        "The reviewed discovery ledger records the multi-agent primary-source audit. It is a deterministic",
        "verification pilot, not a claim that an independent model recovered a hidden holdout set.",
        "Watchlist PDFs are not downloaded or sent to OCR.",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(memo), encoding="utf-8")
    print(f"Built {manifest_path}: 18 accepted + 4 watchlist; duplicate guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
