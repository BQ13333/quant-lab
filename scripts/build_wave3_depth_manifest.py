#!/usr/bin/env python3
"""Build the Wave 3 depth acquisition manifest for seven underweight specialists."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-WAVE3-DEPTH-20260712"
OUTPUT = BATCH / "candidate_manifest.csv"
CORPUS = ROOT / "PhD_Math_Agent_Library/indexes/corpus_index.csv"
FIRST = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-FIRST-LANES-20260712/candidate_manifest.csv"
WAVE2 = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-WAVE2-FOUNDATIONS-20260712/candidate_manifest.csv"
EXISTING_MANIFESTS = (CORPUS, FIRST, WAVE2)
FIELDS = [
    "candidate_id", "lane", "title", "authors", "year", "venue", "landing_url", "pdf_url",
    "doi", "arxiv_id", "license", "peer_review_status", "primary_specialist",
    "secondary_specialists", "tier", "quant_relevance", "selection_status", "selection_reason",
]
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+?)(?:\.pdf)?(?:[?#]|$)", re.IGNORECASE)


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def norm_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value.strip())
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def arxiv_from(value: str) -> str:
    match = ARXIV_RE.search(value or "")
    if not match:
        return ""
    return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE).lower()


def p(
    cid: str,
    lane: str,
    title: str,
    authors: str,
    year: int,
    venue: str,
    landing: str,
    pdf: str,
    primary: str,
    secondary: str,
    tier: str,
    relevance: str,
    *,
    arxiv: str = "",
    doi: str = "",
    license_: str = "arXiv distribution copy; reuse terms require review",
    peer_review_status: str = "peer-reviewed",
) -> dict[str, str]:
    return {
        "candidate_id": cid,
        "lane": lane,
        "title": title,
        "authors": authors,
        "year": str(year),
        "venue": venue,
        "landing_url": landing,
        "pdf_url": pdf,
        "doi": doi,
        "arxiv_id": arxiv,
        "license": license_,
        "peer_review_status": peer_review_status,
        "primary_specialist": primary,
        "secondary_specialists": secondary,
        "tier": tier,
        "quant_relevance": relevance,
        "selection_status": "accepted",
        "selection_reason": (
            "Wave 3 depth review: mathematical rigor, quant/AI relevance, stable open PDF, "
            "and no canonical, Wave 1, or Wave 2 duplicate."
        ),
    }


PAPERS = [
    # Agent 09 — Bayesian inference and latent-variable models.
    p("W3-BAY-01", "bayesian_latent_variable_depth", "MCMC using Hamiltonian dynamics", "Radford M. Neal", 2012, "Handbook of Markov Chain Monte Carlo", "https://arxiv.org/abs/1206.1901", "https://arxiv.org/pdf/1206.1901", "bayesian_latent_variable_specialist", "measure_probability_martingale_specialist;optimization_information_geometry_specialist", "canonical", "Hamiltonian geometry, reversibility, integration error, and diagnostics for rigorous posterior sampling of volatility, regime, and parameter models.", arxiv="1206.1901"),
    p("W3-BAY-02", "bayesian_latent_variable_depth", "Auto-Encoding Variational Bayes", "Diederik P. Kingma; Max Welling", 2013, "ICLR 2014", "https://arxiv.org/abs/1312.6114", "https://arxiv.org/pdf/1312.6114", "bayesian_latent_variable_specialist", "neural_dynamics_sequence_specialist;generative_diffusion_schrodinger_specialist", "canonical", "Reparameterized ELBO optimization for amortized latent-state and Bayesian deep models.", arxiv="1312.6114"),
    p("W3-BAY-03", "bayesian_latent_variable_depth", "Weight Uncertainty in Neural Networks", "Charles Blundell; Julien Cornebise; Koray Kavukcuoglu; Daan Wierstra", 2015, "ICML 2015", "https://arxiv.org/abs/1505.05424", "https://arxiv.org/pdf/1505.05424", "bayesian_latent_variable_specialist", "learning_theory_generalization_specialist;stochastic_control_hjb_bsde_specialist", "ml_bridge", "Bayes-by-Backprop supplies epistemic uncertainty for neural forecasts, risk limits, and exploration-aware policies.", arxiv="1505.05424"),
    p("W3-BAY-04", "bayesian_latent_variable_depth", "Structured Inference Networks for Nonlinear State Space Models", "Rahul G. Krishnan; Uri Shalit; David Sontag", 2016, "AAAI 2017", "https://arxiv.org/abs/1609.09869", "https://arxiv.org/pdf/1609.09869", "bayesian_latent_variable_specialist", "time_series_dependence_specialist;neural_dynamics_sequence_specialist", "modern_canonical", "Deep structured variational inference for nonlinear hidden regimes, latent volatility, and state filtering.", arxiv="1609.09869"),
    p("W3-BAY-05", "bayesian_latent_variable_depth", "Neural Processes", "Marta Garnelo; Jonathan Schwarz; Dan Rosenbaum; Fabio Viola; Danilo J. Rezende; S. M. Ali Eslami; Yee Whye Teh", 2018, "arXiv preprint", "https://arxiv.org/abs/1807.01622", "https://arxiv.org/pdf/1807.01622", "bayesian_latent_variable_specialist", "learning_theory_generalization_specialist;neural_dynamics_sequence_specialist", "modern", "Latent stochastic functions and amortized inference for rapid cross-asset and few-shot uncertainty adaptation.", arxiv="1807.01622", peer_review_status="preprint"),
    p("W3-BAY-06", "bayesian_latent_variable_depth", "Normalizing Flows for Probabilistic Modeling and Inference", "George Papamakarios; Eric Nalisnick; Danilo Jimenez Rezende; Shakir Mohamed; Balaji Lakshminarayanan", 2019, "Journal of Machine Learning Research 22 (2021)", "https://arxiv.org/abs/1912.02762", "https://arxiv.org/pdf/1912.02762", "bayesian_latent_variable_specialist", "generative_diffusion_schrodinger_specialist;robust_statistics_extremes_specialist", "canonical", "Expressive invertible densities for posterior approximation, scenario laws, and calibrated tail inference.", arxiv="1912.02762"),

    # Agent 10 — causal identification and invariant learning.
    p("W3-CAU-01", "causal_identification_invariance_depth", "Double/Debiased Machine Learning for Treatment and Causal Parameters", "Victor Chernozhukov; Denis Chetverikov; Mert Demirer; Esther Duflo; Christian Hansen; Whitney Newey; James Robins", 2016, "The Econometrics Journal 21 (2018)", "https://arxiv.org/abs/1608.00060", "https://arxiv.org/pdf/1608.00060", "causal_identification_invariance_specialist", "high_dimensional_statistics_rmt_specialist;selection_bias_validation_specialist", "canonical", "Orthogonal scores and cross-fitting let flexible ML estimate causal effects while controlling nuisance-model bias.", arxiv="1608.00060"),
    p("W3-CAU-02", "causal_identification_invariance_depth", "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests", "Stefan Wager; Susan Athey", 2015, "Journal of the American Statistical Association 113 (2018)", "https://arxiv.org/abs/1510.04342", "https://arxiv.org/pdf/1510.04342", "causal_identification_invariance_specialist", "high_dimensional_statistics_rmt_specialist;robust_statistics_extremes_specialist", "canonical", "Statistically valid heterogeneous effects across assets, sectors, states, and regimes.", arxiv="1510.04342"),
    p("W3-CAU-03", "causal_identification_invariance_depth", "Anchor regression: heterogeneous data meets causality", "Dominik Rothenhäusler; Nicolai Meinshausen; Peter Bühlmann; Jonas Peters", 2018, "JRSS Series B 83 (2021)", "https://arxiv.org/abs/1801.06229", "https://arxiv.org/pdf/1801.06229", "causal_identification_invariance_specialist", "portfolio_dro_optimal_transport_specialist;learning_theory_generalization_specialist", "canonical", "Connects causal robustness, shift interventions, and regularization for regime-stable signals.", arxiv="1801.06229"),
    p("W3-CAU-04", "causal_identification_invariance_depth", "DAGs with NO TEARS: Continuous Optimization for Structure Learning", "Xun Zheng; Bryon Aragam; Pradeep Ravikumar; Eric P. Xing", 2018, "NeurIPS 2018", "https://arxiv.org/abs/1803.01422", "https://arxiv.org/pdf/1803.01422", "causal_identification_invariance_specialist", "optimization_information_geometry_specialist;high_dimensional_statistics_rmt_specialist", "modern_canonical", "Continuous optimization for acyclic causal discovery in high-dimensional feature systems.", arxiv="1803.01422"),
    p("W3-CAU-05", "causal_identification_invariance_depth", "Towards Causal Representation Learning", "Bernhard Schölkopf; Francesco Locatello; Stefan Bauer; Nan Rosemary Ke; Nal Kalchbrenner; Anirudh Goyal; Yoshua Bengio", 2021, "Proceedings of the IEEE 109", "https://arxiv.org/abs/2102.11107", "https://arxiv.org/pdf/2102.11107", "causal_identification_invariance_specialist", "learning_theory_generalization_specialist;neural_dynamics_sequence_specialist", "modern_canonical", "Identifiability foundations for causal neural representations and environment-generalizing factors.", arxiv="2102.11107"),
    p("W3-CAU-06", "causal_identification_invariance_depth", "Causal Discovery with Continuous Additive Noise Models", "Jonas Peters; Joris Mooij; Dominik Janzing; Bernhard Schölkopf", 2013, "Journal of Machine Learning Research 15 (2014)", "https://arxiv.org/abs/1309.6779", "https://arxiv.org/pdf/1309.6779", "causal_identification_invariance_specialist", "time_series_dependence_specialist;learning_theory_generalization_specialist", "canonical", "Nonlinear identifiability theory and algorithms that distinguish directional mechanisms from correlation.", arxiv="1309.6779"),

    # Agent 11 — sequential inference and online adaptation.
    p("W3-SEQ-01", "sequential_online_adaptation_depth", "Online Convex Programming and Generalized Infinitesimal Gradient Ascent", "Martin Zinkevich", 2003, "ICML 2003", "https://aaai.org/papers/icml03-120-online-convex-programming-and-generalized-infinitesimal-gradient-ascent/", "https://cdn.aaai.org/ICML/2003/ICML03-120.pdf", "sequential_inference_online_learning_specialist", "optimization_information_geometry_specialist;portfolio_dro_optimal_transport_specialist", "canonical", "Foundational online-gradient regret analysis for sequential portfolio, execution, and allocation decisions.", license_="public AAAI proceedings copy; reuse terms require review"),
    p("W3-SEQ-02", "sequential_online_adaptation_depth", "Non-stationary Stochastic Optimization", "Omar Besbes; Yonatan Gur; Assaf Zeevi", 2013, "Operations Research 63 (2015)", "https://arxiv.org/abs/1307.5449", "https://arxiv.org/pdf/1307.5449", "sequential_inference_online_learning_specialist", "optimization_information_geometry_specialist;time_series_dependence_specialist", "canonical", "Variation-budget limits and optimal policies under drifting objectives for adaptive quant research.", arxiv="1307.5449"),
    p("W3-SEQ-03", "sequential_online_adaptation_depth", "Optimal Exploration-Exploitation in a Multi-Armed-Bandit Problem with Non-stationary Rewards", "Omar Besbes; Yonatan Gur; Assaf Zeevi", 2014, "Stochastic Systems 9 (2019)", "https://arxiv.org/abs/1405.3316", "https://arxiv.org/pdf/1405.3316", "sequential_inference_online_learning_specialist", "stochastic_control_hjb_bsde_specialist;time_series_dependence_specialist", "canonical", "Regret-optimal bandit adaptation for signals and strategies whose rewards evolve over time.", arxiv="1405.3316"),
    p("W3-SEQ-04", "sequential_online_adaptation_depth", "Online Learning with Predictable Sequences", "Alexander Rakhlin; Karthik Sridharan", 2012, "COLT / PMLR 30 (2013)", "https://arxiv.org/abs/1208.3728", "https://arxiv.org/pdf/1208.3728", "sequential_inference_online_learning_specialist", "learning_theory_generalization_specialist;optimization_information_geometry_specialist", "canonical", "Optimistic online learning formalizes how predictable market structure can improve regret without losing robustness.", arxiv="1208.3728"),
    p("W3-SEQ-05", "sequential_online_adaptation_depth", "Corralling a Band of Bandit Algorithms", "Alekh Agarwal; Haipeng Luo; Behnam Neyshabur; Robert E. Schapire", 2016, "COLT / PMLR 65 (2017)", "https://arxiv.org/abs/1612.06246", "https://arxiv.org/pdf/1612.06246", "sequential_inference_online_learning_specialist", "stochastic_control_hjb_bsde_specialist;selection_bias_validation_specialist", "modern_canonical", "Meta-algorithm for safely combining competing adaptive learners and specialist trading policies.", arxiv="1612.06246"),
    p("W3-SEQ-06", "sequential_online_adaptation_depth", "Online Meta-Learning", "Chelsea Finn; Aravind Rajeswaran; Sham Kakade; Sergey Levine", 2019, "ICML / PMLR 97", "https://arxiv.org/abs/1902.08438", "https://arxiv.org/pdf/1902.08438", "sequential_inference_online_learning_specialist", "learning_theory_generalization_specialist;neural_dynamics_sequence_specialist", "modern", "Rapid continual adaptation across changing tasks, assets, and market environments.", arxiv="1902.08438"),

    # Agent 14 — neural dynamics and structured sequences.
    p("W3-NDS-01", "neural_dynamics_sequence_depth", "Latent ODEs for Irregularly-Sampled Time Series", "Yulia Rubanova; Ricky T. Q. Chen; David Duvenaud", 2019, "NeurIPS 2019", "https://arxiv.org/abs/1907.03907", "https://arxiv.org/pdf/1907.03907", "neural_dynamics_sequence_specialist", "bayesian_latent_variable_specialist;time_series_dependence_specialist", "modern_canonical", "Continuous latent dynamics for asynchronous events, missing timestamps, and irregular market observations.", arxiv="1907.03907"),
    p("W3-NDS-02", "neural_dynamics_sequence_depth", "Recurrent Neural Networks for Multivariate Time Series with Missing Values", "Zhengping Che; Sanjay Purushotham; Kyunghyun Cho; David Sontag; Yan Liu", 2016, "Scientific Reports 8 (2018)", "https://arxiv.org/abs/1606.01865", "https://arxiv.org/pdf/1606.01865", "neural_dynamics_sequence_specialist", "time_series_dependence_specialist;high_frequency_econometrics_specialist", "canonical", "GRU-D explicitly models informative missingness and decay for stale quotes and asynchronous features.", arxiv="1606.01865"),
    p("W3-NDS-03", "neural_dynamics_sequence_depth", "Deep Kalman Filters", "Rahul G. Krishnan; Uri Shalit; David Sontag", 2015, "arXiv preprint", "https://arxiv.org/abs/1511.05121", "https://arxiv.org/pdf/1511.05121", "neural_dynamics_sequence_specialist", "bayesian_latent_variable_specialist;time_series_dependence_specialist", "core", "Nonlinear neural transitions with probabilistic state-space inference for regime and latent-factor filtering.", arxiv="1511.05121", peer_review_status="preprint"),
    p("W3-NDS-04", "neural_dynamics_sequence_depth", "A Recurrent Latent Variable Model for Sequential Data", "Junyoung Chung; Kyle Kastner; Laurent Dinh; Kratarth Goel; Aaron Courville; Yoshua Bengio", 2015, "NeurIPS 2015", "https://arxiv.org/abs/1506.02216", "https://arxiv.org/pdf/1506.02216", "neural_dynamics_sequence_specialist", "bayesian_latent_variable_specialist;generative_diffusion_schrodinger_specialist", "canonical", "Stochastic hidden-state sequence modeling rather than point-only recurrent forecasts.", arxiv="1506.02216"),
    p("W3-NDS-05", "neural_dynamics_sequence_depth", "Lipschitz Recurrent Neural Networks", "N. Benjamin Erichson; Omri Azencot; Alejandro Queiruga; Liam Hodgkinson; Michael W. Mahoney", 2020, "ICLR 2021", "https://arxiv.org/abs/2006.12070", "https://arxiv.org/pdf/2006.12070", "neural_dynamics_sequence_specialist", "learning_theory_generalization_specialist;optimization_information_geometry_specialist", "theory", "Stability-constrained recurrent dynamics address exploding gradients, long horizons, and perturbation sensitivity.", arxiv="2006.12070"),
    p("W3-NDS-06", "neural_dynamics_sequence_depth", "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", "Yuqi Nie; Nam H. Nguyen; Phanwadee Sinthong; Jayant Kalagnanam", 2022, "ICLR 2023", "https://arxiv.org/abs/2211.14730", "https://arxiv.org/pdf/2211.14730", "neural_dynamics_sequence_specialist", "time_series_dependence_specialist;learning_theory_generalization_specialist", "modern", "PatchTST supplies a strong long-horizon baseline and channel-independent sequence representation.", arxiv="2211.14730"),

    # Agent 15 — neural operators and scientific machine learning.
    p("W3-NOP-01", "neural_operator_sciml_depth", "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators", "Lu Lu; Pengzhan Jin; George Em Karniadakis", 2019, "Nature Machine Intelligence 3 (2021)", "https://arxiv.org/abs/1910.03193", "https://arxiv.org/pdf/1910.03193", "neural_operator_scientific_ml_specialist", "numerical_pde_sde_monte_carlo_specialist;learning_theory_generalization_specialist", "canonical", "Foundational branch-trunk operator architecture backed by an operator universal-approximation theorem.", arxiv="1910.03193"),
    p("W3-NOP-02", "neural_operator_sciml_depth", "On universal approximation and error bounds for Fourier Neural Operators", "Nikola Kovachki; Samuel Lanthaler; Siddhartha Mishra", 2021, "Journal of Machine Learning Research 22 (2021)", "https://arxiv.org/abs/2107.07562", "https://arxiv.org/pdf/2107.07562", "neural_operator_scientific_ml_specialist", "numerical_pde_sde_monte_carlo_specialist;learning_theory_generalization_specialist", "theory", "Approximation theory needed to distinguish FNO capability from benchmark performance.", arxiv="2107.07562"),
    p("W3-NOP-03", "neural_operator_sciml_depth", "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets", "Sifan Wang; Hanwen Wang; Paris Perdikaris", 2021, "Science Advances 7", "https://arxiv.org/abs/2103.10974", "https://arxiv.org/pdf/2103.10974", "neural_operator_scientific_ml_specialist", "numerical_pde_sde_monte_carlo_specialist;stochastic_control_hjb_bsde_specialist", "modern_canonical", "PDE-constrained reusable parameter-to-solution maps for pricing and calibration.", arxiv="2103.10974"),
    p("W3-NOP-04", "neural_operator_sciml_depth", "Solving high-dimensional partial differential equations using deep learning", "Jiequn Han; Arnulf Jentzen; Weinan E", 2017, "Proceedings of the National Academy of Sciences 115 (2018)", "https://arxiv.org/abs/1707.02568", "https://arxiv.org/pdf/1707.02568", "neural_operator_scientific_ml_specialist", "numerical_pde_sde_monte_carlo_specialist;stochastic_control_hjb_bsde_specialist", "canonical", "Seminal stochastic-control and BSDE formulation for high-dimensional derivatives and risk PDEs.", arxiv="1707.02568"),
    p("W3-NOP-05", "neural_operator_sciml_depth", "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers", "Phillip Lippe; Bastiaan S. Veeling; Paris Perdikaris; Richard E. Turner; Johannes Brandstetter", 2023, "NeurIPS 2023", "https://arxiv.org/abs/2308.05732", "https://arxiv.org/pdf/2308.05732", "neural_operator_scientific_ml_specialist", "neural_dynamics_sequence_specialist;numerical_pde_sde_monte_carlo_specialist", "modern", "Controls accumulated rollout error and stability when learned operators drive multi-step scenarios.", arxiv="2308.05732"),
    p("W3-NOP-06", "neural_operator_sciml_depth", "Wavelet neural operator: a neural operator for parametric partial differential equations", "Tapas Tripura; Souvik Chakraborty", 2022, "Computer Methods in Applied Mechanics and Engineering 404 (2023)", "https://arxiv.org/abs/2205.02191", "https://arxiv.org/pdf/2205.02191", "neural_operator_scientific_ml_specialist", "time_series_dependence_specialist;high_frequency_econometrics_specialist", "modern", "Localized multiscale operator learning for rough, transient, and frequency-varying financial dynamics.", arxiv="2205.02191"),

    # Agent 16 — generative diffusion and Schrodinger bridges.
    p("W3-GEN-01", "generative_diffusion_schrodinger_depth", "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", "Jascha Sohl-Dickstein; Eric A. Weiss; Niru Maheswaranathan; Surya Ganguli", 2015, "ICML / PMLR 37", "https://arxiv.org/abs/1503.03585", "https://arxiv.org/pdf/1503.03585", "generative_diffusion_schrodinger_specialist", "measure_probability_martingale_specialist;bayesian_latent_variable_specialist", "canonical", "Original diffusion construction connecting forward entropy production and learned reverse dynamics.", arxiv="1503.03585"),
    p("W3-GEN-02", "generative_diffusion_schrodinger_depth", "Denoising Diffusion Probabilistic Models", "Jonathan Ho; Ajay Jain; Pieter Abbeel", 2020, "NeurIPS 2020", "https://arxiv.org/abs/2006.11239", "https://arxiv.org/pdf/2006.11239", "generative_diffusion_schrodinger_specialist", "bayesian_latent_variable_specialist;robust_statistics_extremes_specialist", "canonical", "Canonical DDPM objective and parameterization complementing the existing Improved DDPM paper.", arxiv="2006.11239"),
    p("W3-GEN-03", "generative_diffusion_schrodinger_depth", "Flow Matching for Generative Modeling", "Yaron Lipman; Ricky T. Q. Chen; Heli Ben-Hamu; Maximilian Nickel; Matt Le", 2022, "ICLR 2023", "https://arxiv.org/abs/2210.02747", "https://arxiv.org/pdf/2210.02747", "generative_diffusion_schrodinger_specialist", "optimization_information_geometry_specialist;stochastic_calculus_rough_paths_specialist", "modern_canonical", "Simulation-free continuous normalizing-flow training for conditional path and stress-law generation.", arxiv="2210.02747"),
    p("W3-GEN-04", "generative_diffusion_schrodinger_depth", "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions", "Michael S. Albergo; Nicholas M. Boffi; Eric Vanden-Eijnden", 2023, "arXiv preprint", "https://arxiv.org/abs/2303.08797", "https://arxiv.org/pdf/2303.08797", "generative_diffusion_schrodinger_specialist", "stochastic_calculus_rough_paths_specialist;optimization_information_geometry_specialist", "theory", "Mathematical bridge among deterministic flows, stochastic diffusions, scores, and transport objectives.", arxiv="2303.08797", peer_review_status="preprint"),
    p("W3-GEN-05", "generative_diffusion_schrodinger_depth", "Elucidating the Design Space of Diffusion-Based Generative Models", "Tero Karras; Miika Aittala; Timo Aila; Samuli Laine", 2022, "NeurIPS 2022", "https://arxiv.org/abs/2206.00364", "https://arxiv.org/pdf/2206.00364", "generative_diffusion_schrodinger_specialist", "numerical_pde_sde_monte_carlo_specialist;optimization_information_geometry_specialist", "modern", "Separates noise schedules, parameterization, preconditioning, and sampling for controlled model design.", arxiv="2206.00364"),
    p("W3-GEN-06", "generative_diffusion_schrodinger_depth", "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", "Yusuke Tashiro; Jiaming Song; Yang Song; Stefano Ermon", 2021, "NeurIPS 2021", "https://arxiv.org/abs/2107.03502", "https://arxiv.org/pdf/2107.03502", "generative_diffusion_schrodinger_specialist", "neural_dynamics_sequence_specialist;time_series_dependence_specialist", "modern", "Conditional diffusion for missing multivariate sequences and sparse or asynchronous market panels.", arxiv="2107.03502"),

    # Agent 17 — graph learning and systemic risk.
    p("W3-GRA-01", "graph_network_systemic_risk_depth", "Semi-Supervised Classification with Graph Convolutional Networks", "Thomas N. Kipf; Max Welling", 2016, "ICLR 2017", "https://arxiv.org/abs/1609.02907", "https://arxiv.org/pdf/1609.02907", "graph_network_systemic_risk_specialist", "high_dimensional_statistics_rmt_specialist;learning_theory_generalization_specialist", "canonical", "Foundational spectral-to-message-passing derivation for learned asset and counterparty graphs.", arxiv="1609.02907"),
    p("W3-GRA-02", "graph_network_systemic_risk_depth", "Graph Attention Networks", "Petar Velickovic; Guillem Cucurull; Arantxa Casanova; Adriana Romero; Pietro Lio; Yoshua Bengio", 2017, "ICLR 2018", "https://arxiv.org/abs/1710.10903", "https://arxiv.org/pdf/1710.10903", "graph_network_systemic_risk_specialist", "high_dimensional_statistics_rmt_specialist;neural_dynamics_sequence_specialist", "canonical", "Learns heterogeneous neighbor weights for time-varying transmission and exposure importance.", arxiv="1710.10903"),
    p("W3-GRA-03", "graph_network_systemic_risk_depth", "How Powerful are Graph Neural Networks?", "Keyulu Xu; Weihua Hu; Jure Leskovec; Stefanie Jegelka", 2018, "ICLR 2019", "https://arxiv.org/abs/1810.00826", "https://arxiv.org/pdf/1810.00826", "graph_network_systemic_risk_specialist", "learning_theory_generalization_specialist;high_dimensional_statistics_rmt_specialist", "theory", "Message-passing expressivity through Weisfeiler-Lehman theory and the Graph Isomorphism Network.", arxiv="1810.00826"),
    p("W3-GRA-04", "graph_network_systemic_risk_depth", "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", "Aldo Pareja; Giacomo Domeniconi; Jie Chen; Tengfei Ma; Toyotaro Suzumura; Hiroki Kanezashi; Tim Kaler; Tao B. Schardl; Charles E. Leiserson", 2019, "AAAI 2020", "https://arxiv.org/abs/1902.10191", "https://arxiv.org/pdf/1902.10191", "graph_network_systemic_risk_specialist", "neural_dynamics_sequence_specialist;time_series_dependence_specialist", "modern", "Evolves graph parameters through changing ownership, correlation, liquidity, and counterparty networks.", arxiv="1902.10191"),
    p("W3-GRA-05", "graph_network_systemic_risk_depth", "Systemic Risk in Financial Systems: Properties of Equilibria", "John Stachurski", 2022, "arXiv preprint", "https://arxiv.org/abs/2202.11183", "https://arxiv.org/pdf/2202.11183", "graph_network_systemic_risk_specialist", "measure_probability_martingale_specialist;optimization_information_geometry_specialist", "theory", "Fixed-point and uniqueness analysis sharpening the mathematical basis of clearing-network stress models.", arxiv="2202.11183", peer_review_status="preprint"),
    p("W3-GRA-06", "graph_network_systemic_risk_depth", "Systemic Risk and Stability in Financial Networks", "Daron Acemoglu; Asuman Ozdaglar; Alireza Tahbaz-Salehi", 2013, "NBER Working Paper 18727 / American Economic Review 105 (2015)", "https://hdl.handle.net/1721.1/77606", "https://dspace.mit.edu/server/api/core/bitstreams/acfa6ad7-ac9f-4441-8d28-dc1f8f8b4491/content", "graph_network_systemic_risk_specialist", "measure_probability_martingale_specialist;portfolio_dro_optimal_transport_specialist", "canonical", "Topology-dependent contagion phase transitions and network externalities linking graph structure to fragility.", doi="10.3386/w18727", license_="MIT repository working-paper copy; reuse terms require review"),
]


def load_existing() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    titles: dict[str, str] = {}
    urls: dict[str, str] = {}
    arxiv_ids: dict[str, str] = {}
    for path in EXISTING_MANIFESTS:
        if not path.is_file():
            raise SystemExit(f"Required duplicate-guard input is missing: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                owner = row.get("document_id") or row.get("candidate_id") or path.name
                title = norm(row.get("title", ""))
                if title:
                    titles[title] = owner
                for key in ("source_url", "landing_url", "pdf_url"):
                    raw_url = row.get(key, "")
                    url = norm_url(raw_url)
                    if url:
                        urls[url] = owner
                    source_arxiv = arxiv_from(raw_url)
                    if source_arxiv:
                        arxiv_ids[source_arxiv] = owner
                explicit_arxiv = re.sub(r"v\d+$", "", row.get("arxiv_id", ""), flags=re.IGNORECASE).lower()
                if explicit_arxiv:
                    arxiv_ids[explicit_arxiv] = owner
    return titles, urls, arxiv_ids


def main() -> None:
    if len(PAPERS) != 42:
        raise SystemExit(f"Expected 42 papers, found {len(PAPERS)}")

    for key in ("candidate_id", "title", "landing_url", "pdf_url"):
        values = [norm(row[key]) if key == "title" else norm_url(row[key]) if key.endswith("url") else row[key] for row in PAPERS]
        if len(values) != len(set(values)):
            raise SystemExit(f"Internal duplicate {key}")
    arxiv_ids = [row["arxiv_id"].lower() for row in PAPERS if row["arxiv_id"]]
    if len(arxiv_ids) != len(set(arxiv_ids)):
        raise SystemExit("Internal duplicate arXiv ID")

    existing_titles, existing_urls, existing_arxiv = load_existing()
    collisions: list[str] = []
    for row in PAPERS:
        cid = row["candidate_id"]
        title = norm(row["title"])
        if title in existing_titles:
            collisions.append(f"{cid}: title -> {existing_titles[title]}")
        for key in ("landing_url", "pdf_url"):
            url = norm_url(row[key])
            if url in existing_urls:
                collisions.append(f"{cid}: {key} -> {existing_urls[url]}")
        arxiv_id = row["arxiv_id"].lower()
        if arxiv_id and arxiv_id in existing_arxiv:
            collisions.append(f"{cid}: arXiv {arxiv_id} -> {existing_arxiv[arxiv_id]}")
    if collisions:
        raise SystemExit("Existing acquisition collisions:\n" + "\n".join(collisions))

    BATCH.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(PAPERS)
    print(f"Wrote {len(PAPERS)} candidates to {OUTPUT}")


if __name__ == "__main__":
    main()
