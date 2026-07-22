#!/usr/bin/env python3
"""Build the acquisition-ready Wave 4 cross-specialist synthesis manifest.

This builder deliberately does not download PDFs or mutate canonical indexes.  It
fails closed if a proposed title or arXiv source identifier already exists in the
canonical corpus, and it validates every specialist route against the live roster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-WAVE4-SYNTHESIS-20260712"
CORPUS = ROOT / "PhD_Math_Agent_Library/indexes/corpus_index.csv"
ROSTER = ROOT / "PhD_Math_Agent_Library/indexes/agent_roster.csv"
WAVE3 = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-WAVE3-DEPTH-20260712/candidate_manifest.csv"
FIELDS = [
    "candidate_id", "lane", "title", "authors", "year", "venue", "landing_url", "pdf_url",
    "doi", "arxiv_id", "license", "peer_review_status", "primary_specialist",
    "secondary_specialists", "tier", "quant_relevance", "selection_status", "selection_reason",
]
ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")


def norm(value: str) -> str:
    """Normalize scholarly titles for deterministic duplicate checks."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def p(
    cid: str,
    lane: str,
    title: str,
    authors: str,
    year: int,
    arxiv_id: str,
    primary: str,
    secondary: str,
    tier: str,
    rationale: str,
) -> dict[str, str]:
    return {
        "candidate_id": cid,
        "lane": lane,
        "title": title,
        "authors": authors,
        "year": str(year),
        "venue": "arXiv",
        "landing_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "doi": "",
        "arxiv_id": arxiv_id,
        "license": "arXiv distribution copy; reuse terms require review",
        "peer_review_status": "research preprint; publication status not independently asserted",
        "primary_specialist": primary,
        "secondary_specialists": secondary,
        "tier": tier,
        "quant_relevance": rationale,
        "selection_status": "accepted",
        "selection_reason": (
            "Wave 4 cross-specialist review: theorem depth, explicit math/AI/quant synthesis, "
            "direct open PDF, and no canonical-corpus title or arXiv-ID duplicate."
        ),
    }


PAPERS = [
    # Lane 1 — arbitrage-safe neural market worlds.
    p("W4-01", "arbitrage_safe_neural_market_worlds", "Arbitrage-free neural-SDE market models", "Samuel N. Cohen; Christoph Reisinger; Sheng Wang", 2021, "2105.11053", "generative_diffusion_schrodinger_specialist", "measure_probability_martingale_specialist;stochastic_calculus_rough_paths_specialist;stochastic_control_hjb_bsde_specialist;neural_operator_scientific_ml_specialist", "modern_canonical", "Constrained neural-SDE state spaces that preserve option-book static-arbitrage inequalities."),
    p("W4-02", "arbitrage_safe_neural_market_worlds", "Robust pricing and hedging via neural SDEs", "Patryk Gierjatowicz; Marc Sabate-Vidales; David Šiška; Lukasz Szpruch; Žan Žurič", 2020, "2007.04154", "stochastic_control_hjb_bsde_specialist", "stochastic_calculus_rough_paths_specialist;robust_statistics_extremes_specialist;generative_diffusion_schrodinger_specialist;portfolio_dro_optimal_transport_specialist", "modern_canonical", "Adversarial path-law learning connected directly to robust derivative pricing and hedge construction."),
    p("W4-03", "arbitrage_safe_neural_market_worlds", "Hedging option books using neural-SDE market models", "Samuel N. Cohen; Christoph Reisinger; Sheng Wang", 2022, "2205.15991", "stochastic_control_hjb_bsde_specialist", "stochastic_calculus_rough_paths_specialist;generative_diffusion_schrodinger_specialist;portfolio_dro_optimal_transport_specialist", "modern", "Arbitrage-preserving learned dynamics converted into book-level hedging decisions."),
    p("W4-04", "arbitrage_safe_neural_market_worlds", "Estimating risks of option books using neural-SDE market models", "Samuel N. Cohen; Christoph Reisinger; Sheng Wang", 2022, "2202.07148", "robust_statistics_extremes_specialist", "stochastic_calculus_rough_paths_specialist;bayesian_latent_variable_specialist;generative_diffusion_schrodinger_specialist;portfolio_dro_optimal_transport_specialist", "modern", "Neural market simulation made operational for option-book tail and model-risk measurement."),
    p("W4-05", "arbitrage_safe_neural_market_worlds", "Non-adversarial training of Neural SDEs with signature kernel scores", "Zacharia Issa; Blanka Horvath; Maud Lemercier; Cristopher Salvi", 2023, "2305.16274", "generative_diffusion_schrodinger_specialist", "stochastic_calculus_rough_paths_specialist;bayesian_latent_variable_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "modern", "Mathematically grounded signature-kernel scoring rules as a stable alternative to adversarial training."),
    p("W4-06", "arbitrage_safe_neural_market_worlds", "Robust financial calibration: a Bayesian approach for neural SDEs", "Christa Cuchiero; Eva Flonner; Kevin Kurt", 2024, "2409.06551", "bayesian_latent_variable_specialist", "stochastic_calculus_rough_paths_specialist;robust_statistics_extremes_specialist;generative_diffusion_schrodinger_specialist;optimization_information_geometry_specialist", "modern", "Posterior calibration uncertainty and robustness for learned financial dynamics."),

    # Lane 2 — causal alpha under regime shift and adaptive research.
    p("W4-07", "causal_alpha_regime_shift", "Invariant Models for Causal Transfer Learning", "Mateo Rojas-Carulla; Bernhard Schölkopf; Richard Turner; Jonas Peters", 2015, "1507.05333", "causal_identification_invariance_specialist", "learning_theory_generalization_specialist;time_series_dependence_specialist;selection_bias_validation_specialist;portfolio_dro_optimal_transport_specialist", "canonical", "Adversarially optimal invariant predictors for transfer across unseen market regimes."),
    p("W4-08", "causal_alpha_regime_shift", "Causal Dantzig: fast inference in linear structural equation models with hidden variables under additive interventions", "Dominik Rothenhäusler; Peter Bühlmann; Nicolai Meinshausen", 2017, "1706.06159", "causal_identification_invariance_specialist", "high_dimensional_statistics_rmt_specialist;robust_statistics_extremes_specialist;learning_theory_generalization_specialist;selection_bias_validation_specialist", "canonical", "High-dimensional causal inference across environments with hidden confounding, finite-sample bounds, and predictive guarantees."),
    p("W4-09", "causal_alpha_regime_shift", "A Unifying Causal Framework for Analyzing Dataset Shift-stable Learning Algorithms", "Adarsh Subbaswamy; Bryant Chen; Suchi Saria", 2019, "1905.11374", "causal_identification_invariance_specialist", "sequential_inference_online_learning_specialist;learning_theory_generalization_specialist;selection_bias_validation_specialist;time_series_dependence_specialist", "canonical", "A causal criterion for distinguishing genuinely shift-stable predictors from empirical robustness stories."),
    p("W4-10", "causal_alpha_regime_shift", "Out-of-Distribution Generalization via Risk Extrapolation (REx)", "David Krueger; Ethan Caballero; Joern-Henrik Jacobsen; Amy Zhang; Jonathan Binas; Dinghuai Zhang; Remi Le Priol; Aaron Courville", 2020, "2003.00688", "causal_identification_invariance_specialist", "robust_statistics_extremes_specialist;sequential_inference_online_learning_specialist;learning_theory_generalization_specialist;portfolio_dro_optimal_transport_specialist", "modern_canonical", "Environment invariance turned into an extrapolative optimization objective for regime robustness."),
    p("W4-11", "causal_alpha_regime_shift", "Causal Bandits: Learning Good Interventions via Causal Inference", "Finnian Lattimore; Tor Lattimore; Mark D. Reid", 2016, "1606.03203", "sequential_inference_online_learning_specialist", "causal_identification_invariance_specialist;stochastic_control_hjb_bsde_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "canonical", "Causal graph structure used to obtain provably lower online intervention regret."),
    p("W4-12", "causal_alpha_regime_shift", "Anytime-Valid Linear Models and Regression Adjusted Causal Inference in Randomized Experiments", "Michael Lindon; Dae Woong Ham; Martin Tingley; Iavor Bojinov", 2022, "2210.08589", "sequential_inference_online_learning_specialist", "causal_identification_invariance_specialist;selection_bias_validation_specialist;learning_theory_generalization_specialist;measure_probability_martingale_specialist", "modern", "Confidence sequences and optional-stopping-safe causal inference for adaptive quant experiments."),

    # Lane 3 — robust Bayesian tails and decision uncertainty.
    p("W4-13", "robust_bayesian_tail_uncertainty", "A General Framework for Updating Belief Distributions", "Pier Giovanni Bissiri; Chris Holmes; Stephen Walker", 2013, "1306.6430", "bayesian_latent_variable_specialist", "robust_statistics_extremes_specialist;optimization_information_geometry_specialist;portfolio_dro_optimal_transport_specialist;learning_theory_generalization_specialist", "canonical", "Coherent generalized Bayesian updating from loss functions rather than a presumed-correct likelihood."),
    p("W4-14", "robust_bayesian_tail_uncertainty", "Robust Bayesian inference via coarsening", "Jeffrey W. Miller; David B. Dunson", 2015, "1506.06101", "bayesian_latent_variable_specialist", "robust_statistics_extremes_specialist;time_series_dependence_specialist;learning_theory_generalization_specialist;selection_bias_validation_specialist", "canonical", "Posterior decisions protected against small but consequential model misspecification."),
    p("W4-15", "robust_bayesian_tail_uncertainty", "Generalized Variational Inference: Three arguments for deriving new Posteriors", "Jeremias Knoblauch; Jack Jewson; Theodoros Damoulas", 2019, "1904.02063", "bayesian_latent_variable_specialist", "robust_statistics_extremes_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "modern_canonical", "Robust losses, divergences, and variational families combined in one principled inference objective."),
    p("W4-16", "robust_bayesian_tail_uncertainty", "A General Method for Robust Bayesian Modeling", "Chong Wang; David M. Blei", 2015, "1510.05078", "bayesian_latent_variable_specialist", "robust_statistics_extremes_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "canonical", "A reusable construction for robustifying existing Bayesian and latent-variable models."),
    p("W4-17", "robust_bayesian_tail_uncertainty", "Scalable semiparametric inference for the means of heavy-tailed distributions", "Matt Taddy; Hedibert Freitas Lopes; Matt Gardner", 2016, "1602.08066", "robust_statistics_extremes_specialist", "bayesian_latent_variable_specialist;high_dimensional_statistics_rmt_specialist;learning_theory_generalization_specialist;selection_bias_validation_specialist", "canonical", "Scalable semiparametric inference when returns or losses have extreme non-Gaussian tails."),
    p("W4-18", "robust_bayesian_tail_uncertainty", "Conformal Risk Control", "Anastasios N. Angelopoulos; Stephen Bates; Adam Fisch; Lihua Lei; Tal Schuster", 2022, "2208.02814", "sequential_inference_online_learning_specialist", "robust_statistics_extremes_specialist;selection_bias_validation_specialist;learning_theory_generalization_specialist;portfolio_dro_optimal_transport_specialist", "modern_canonical", "Finite-sample control of user-defined portfolio or trading losses from predictive uncertainty."),

    # Lane 4 — microstructure-aware execution and adaptive control.
    p("W4-19", "microstructure_adaptive_execution", "Optimal order placement in limit order markets", "Rama Cont; Arseniy Kukanov", 2012, "1210.1625", "microstructure_hawkes_execution_specialist", "stochastic_control_hjb_bsde_specialist;high_frequency_econometrics_specialist;optimization_information_geometry_specialist;portfolio_dro_optimal_transport_specialist", "canonical", "Convex, queue- and fee-aware routing of limit and market orders across venues."),
    p("W4-20", "microstructure_adaptive_execution", "Optimal Execution with Dynamic Order Flow Imbalance", "Kyle Bechler; Mike Ludkovski", 2014, "1409.2618", "microstructure_hawkes_execution_specialist", "stochastic_control_hjb_bsde_specialist;time_series_dependence_specialist;high_frequency_econometrics_specialist;sequential_inference_online_learning_specialist", "canonical", "Execution control coupled to dynamically learned order-flow information and receding horizons."),
    p("W4-21", "microstructure_adaptive_execution", "Multi-asset optimal execution and statistical arbitrage strategies under Ornstein-Uhlenbeck dynamics", "Philippe Bergault; Fayçal Drissi; Olivier Guéant", 2021, "2103.13773", "microstructure_hawkes_execution_specialist", "stochastic_control_hjb_bsde_specialist;time_series_dependence_specialist;portfolio_dro_optimal_transport_specialist;optimization_information_geometry_specialist", "modern_canonical", "Multivariate mean reversion, matrix Riccati control, execution, and statistical arbitrage in one model."),
    p("W4-22", "microstructure_adaptive_execution", "Optimal Execution Using Reinforcement Learning", "Cong Zheng; Jiafa He; Can Yang", 2023, "2306.17178", "microstructure_hawkes_execution_specialist", "stochastic_control_hjb_bsde_specialist;sequential_inference_online_learning_specialist;neural_dynamics_sequence_specialist;selection_bias_validation_specialist", "modern", "Cross-venue information in execution policies, enabling an RL-versus-control benchmark."),
    p("W4-23", "microstructure_adaptive_execution", "Passive Market Impact: A Point Process Approach", "Youssef Ouazzani Chahdi; Mathieu Rosenbaum; Grégoire Szymanski", 2024, "2412.07461", "microstructure_hawkes_execution_specialist", "high_frequency_econometrics_specialist;time_series_dependence_specialist;graph_network_systemic_risk_specialist;robust_statistics_extremes_specialist", "modern", "Point-process foundations for the impact generated by passive liquidity provision."),
    p("W4-24", "microstructure_adaptive_execution", "Reinforcement Learning for Trade Execution with Market and Limit Orders", "Patrick Cheridito; Moritz Weiss", 2025, "2507.06345", "microstructure_hawkes_execution_specialist", "stochastic_control_hjb_bsde_specialist;sequential_inference_online_learning_specialist;neural_dynamics_sequence_specialist;optimization_information_geometry_specialist", "modern", "Joint market/limit allocation policies learned in a strategic limit-order-book environment."),

    # Lane 5 — certified operator, BSDE, and dynamic-risk engines.
    p("W4-25", "certified_operator_bsde_risk_engines", "A Mathematical Guide to Operator Learning", "Nicolas Boullé; Alex Townsend", 2023, "2312.14688", "neural_operator_scientific_ml_specialist", "numerical_pde_sde_monte_carlo_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist;neural_dynamics_sequence_specialist", "canonical", "Functional-analysis foundations for auditing neural pricing and calibration operators."),
    p("W4-26", "certified_operator_bsde_risk_engines", "Data Complexity Estimates for Operator Learning", "Nikola B. Kovachki; Samuel Lanthaler; Hrushikesh Mhaskar", 2024, "2405.15992", "neural_operator_scientific_ml_specialist", "learning_theory_generalization_specialist;high_dimensional_statistics_rmt_specialist;numerical_pde_sde_monte_carlo_specialist;optimization_information_geometry_specialist", "theory", "Training-data complexity estimates for reliable function-valued risk operators."),
    p("W4-27", "certified_operator_bsde_risk_engines", "Interpolating between BSDEs and PINNs: deep learning for elliptic and parabolic boundary value problems", "Nikolas Nüsken; Lorenz Richter", 2021, "2112.03749", "neural_operator_scientific_ml_specialist", "stochastic_control_hjb_bsde_specialist;numerical_pde_sde_monte_carlo_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "modern_canonical", "A unified continuum between probabilistic BSDE and residual-based PINN solvers."),
    p("W4-28", "certified_operator_bsde_risk_engines", "SIG-BSDE for Dynamic Risk Measures", "Nacira Agram; Jan Rems; Emanuela Rosazza Gianin", 2024, "2408.02853", "stochastic_control_hjb_bsde_specialist", "stochastic_calculus_rough_paths_specialist;numerical_pde_sde_monte_carlo_specialist;robust_statistics_extremes_specialist;neural_operator_scientific_ml_specialist;portfolio_dro_optimal_transport_specialist", "modern", "Path signatures and BSDEs combined for learned, time-consistent nonlinear risk measures."),
    p("W4-29", "certified_operator_bsde_risk_engines", "Deep Signature FBSDE Algorithm", "Qi Feng; Man Luo; Zhaoyu Zhang", 2021, "2108.10504", "neural_operator_scientific_ml_specialist", "stochastic_calculus_rough_paths_specialist;stochastic_control_hjb_bsde_specialist;numerical_pde_sde_monte_carlo_specialist;neural_dynamics_sequence_specialist;optimization_information_geometry_specialist", "modern", "Signature representations joined to forward-backward stochastic solvers for non-Markovian pricing."),
    p("W4-30", "certified_operator_bsde_risk_engines", "Convergence of a Deep BSDE solver with jumps", "Alessandro Gnoatto; Katharina Oberpriller; Athena Picarelli", 2025, "2501.09727", "numerical_pde_sde_monte_carlo_specialist", "stochastic_control_hjb_bsde_specialist;robust_statistics_extremes_specialist;learning_theory_generalization_specialist;neural_operator_scientific_ml_specialist;optimization_information_geometry_specialist", "theory", "Convergence theory for neural pricing and risk solvers under discontinuous jump dynamics."),

    # Lane 6 — multi-period portfolios under costs, uncertainty, and risk-sensitive learning.
    p("W4-31", "multiperiod_portfolio_risk_control", "Large-Scale Portfolio Allocation Under Transaction Costs and Model Uncertainty", "Nikolaus Hautsch; Stefan Voigt", 2017, "1709.06296", "portfolio_dro_optimal_transport_specialist", "high_dimensional_statistics_rmt_specialist;robust_statistics_extremes_specialist;optimization_information_geometry_specialist;microstructure_hawkes_execution_specialist;selection_bias_validation_specialist", "canonical", "High-dimensional covariance uncertainty, regularization, and realistic turnover costs linked together."),
    p("W4-32", "multiperiod_portfolio_risk_control", "Multi-Period Portfolio Optimization using Model Predictive Control with Mean-Variance and Risk Parity Frameworks", "Xiaoyue Li; A. Sinem Uysal; John M. Mulvey", 2021, "2103.10813", "portfolio_dro_optimal_transport_specialist", "stochastic_control_hjb_bsde_specialist;sequential_inference_online_learning_specialist;optimization_information_geometry_specialist;high_dimensional_statistics_rmt_specialist", "modern", "Allocation formulated as a repeatedly updated control problem rather than a static optimizer."),
    p("W4-33", "multiperiod_portfolio_risk_control", "Numerical Solution of Dynamic Portfolio Optimization with Transaction Costs", "Yongyang Cai; Kenneth Judd; Rong Xu", 2020, "2003.01809", "portfolio_dro_optimal_transport_specialist", "stochastic_control_hjb_bsde_specialist;numerical_pde_sde_monte_carlo_specialist;optimization_information_geometry_specialist;microstructure_hawkes_execution_specialist", "canonical", "Numerical dynamic-programming machinery for nonlinear cost-aware portfolio policies."),
    p("W4-34", "multiperiod_portfolio_risk_control", "Pessimism Meets Risk: Risk-Sensitive Offline Reinforcement Learning", "Dake Zhang; Boxiang Lyu; Shuang Qiu; Mladen Kolar; Tong Zhang", 2024, "2407.07631", "stochastic_control_hjb_bsde_specialist", "robust_statistics_extremes_specialist;sequential_inference_online_learning_specialist;learning_theory_generalization_specialist;portfolio_dro_optimal_transport_specialist", "modern", "Distributional risk and offline-policy extrapolation error handled simultaneously."),
    p("W4-35", "multiperiod_portfolio_risk_control", "Risk-Sensitive Reinforcement Learning: Near-Optimal Risk-Sample Tradeoff in Regret", "Yingjie Fei; Zhuoran Yang; Yudong Chen; Zhaoran Wang; Qiaomin Xie", 2020, "2006.13827", "sequential_inference_online_learning_specialist", "stochastic_control_hjb_bsde_specialist;robust_statistics_extremes_specialist;learning_theory_generalization_specialist;optimization_information_geometry_specialist", "theory", "Regret and sample-complexity theory for policies optimized beyond expected reward."),
    p("W4-36", "multiperiod_portfolio_risk_control", "Multi-Period Trading via Convex Optimization", "Stephen Boyd; Enzo Busseti; Steven Diamond; Ronald N. Kahn; Kwangmoo Koh; Peter Nystrup; Jan Speth", 2017, "1705.00109", "portfolio_dro_optimal_transport_specialist", "optimization_information_geometry_specialist;stochastic_control_hjb_bsde_specialist;time_series_dependence_specialist;microstructure_hawkes_execution_specialist;high_dimensional_statistics_rmt_specialist", "canonical", "An implementable convex architecture joining forecasts, risk, constraints, holdings, and transaction costs."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def corpus_arxiv_ids(rows: list[dict[str, str]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for row in rows:
        haystack = " ".join(
            (row.get("source_url", ""), row.get("original_pdf_path", ""), row.get("document_id", ""))
        )
        for match in ARXIV_RE.finditer(haystack):
            found[match.group(1)] = row.get("document_id", "existing")
    return found


def validate_candidates() -> dict[str, object]:
    if len(PAPERS) != 36:
        raise SystemExit(f"Expected 36 Wave 4 candidates, found {len(PAPERS)}")

    for field in ("candidate_id", "title", "arxiv_id", "landing_url", "pdf_url"):
        values = [norm(row[field]) if field == "title" else row[field] for row in PAPERS]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise SystemExit(f"Duplicate Wave 4 {field}: {duplicates}")

    expected_ids = [f"W4-{number:02d}" for number in range(1, 37)]
    actual_ids = [row["candidate_id"] for row in PAPERS]
    if actual_ids != expected_ids:
        raise SystemExit(f"Wave 4 candidate IDs are not the stable W4-01..W4-36 sequence: {actual_ids}")

    lane_counts = Counter(row["lane"] for row in PAPERS)
    if len(lane_counts) != 6 or set(lane_counts.values()) != {6}:
        raise SystemExit(f"Expected six candidates in each of six lanes, found {dict(lane_counts)}")

    roster_rows = read_csv(ROSTER)
    roster_ids = {row["agent_id"] for row in roster_rows}
    route_errors: list[tuple[str, str]] = []
    for row in PAPERS:
        routes = [row["primary_specialist"], *row["secondary_specialists"].split(";")]
        route_errors.extend((row["candidate_id"], route) for route in routes if route not in roster_ids)
    if route_errors:
        raise SystemExit(f"Unknown specialist routes: {route_errors}")

    corpus_rows = read_csv(CORPUS)
    existing_titles = {norm(row["title"]): row.get("document_id", "existing") for row in corpus_rows}
    title_collisions = [
        (row["candidate_id"], existing_titles[norm(row["title"])])
        for row in PAPERS
        if norm(row["title"]) in existing_titles
    ]
    if title_collisions:
        raise SystemExit(f"Canonical-corpus title collisions: {title_collisions}")

    existing_arxiv = corpus_arxiv_ids(corpus_rows)
    arxiv_collisions = [
        (row["candidate_id"], existing_arxiv[row["arxiv_id"]])
        for row in PAPERS
        if row["arxiv_id"] in existing_arxiv
    ]
    if arxiv_collisions:
        raise SystemExit(f"Canonical-corpus arXiv-ID collisions: {arxiv_collisions}")

    wave3_rows = read_csv(WAVE3) if WAVE3.is_file() else []
    wave3_titles = {norm(row["title"]): row["candidate_id"] for row in wave3_rows}
    wave3_arxiv = {row["arxiv_id"]: row["candidate_id"] for row in wave3_rows if row.get("arxiv_id")}
    planned_collisions = [
        (row["candidate_id"], wave3_titles[norm(row["title"])])
        for row in PAPERS if norm(row["title"]) in wave3_titles
    ] + [
        (row["candidate_id"], wave3_arxiv[row["arxiv_id"]])
        for row in PAPERS if row["arxiv_id"] in wave3_arxiv
    ]
    if planned_collisions:
        raise SystemExit(f"Wave 3/Wave 4 planned-acquisition collisions: {planned_collisions}")

    return {
        "status": "pass",
        "mode": "acquisition_ready",
        "candidate_count": len(PAPERS),
        "lane_count": len(lane_counts),
        "lane_counts": dict(sorted(lane_counts.items())),
        "primary_specialist_counts": dict(sorted(Counter(row["primary_specialist"] for row in PAPERS).items())),
        "canonical_corpus_rows_checked": len(corpus_rows),
        "canonical_title_collisions": 0,
        "canonical_arxiv_id_collisions": 0,
        "wave3_rows_checked": len(wave3_rows),
        "wave3_title_or_arxiv_collisions": 0,
        "roster_specialists_checked": len(roster_ids),
        "unknown_specialist_routes": 0,
        "canonical_corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BATCH,
        help=f"Wave 4 acquisition directory (default: {DEFAULT_BATCH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_candidates()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "candidate_manifest.csv"
    report_path = output_dir / "duplicate_guard_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(PAPERS)

    report.update(
        {
            "manifest_relative_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(PAPERS)} acquisition-ready candidates to {manifest_path}")
    print(f"Wrote duplicate-guard report to {report_path}")


if __name__ == "__main__":
    main()
