#!/usr/bin/env python3
"""Build the curated first-lane acquisition manifest for the 20-agent panel."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "PhD_Math_Agent_Library/acquisition_staging/ACQ-FIRST-LANES-20260712"
OUTPUT = BATCH / "candidate_manifest.csv"
CORPUS_INDEX = ROOT / "PhD_Math_Agent_Library/indexes/corpus_index.csv"

FIELDS = [
    "candidate_id", "lane", "title", "authors", "year", "venue", "landing_url",
    "pdf_url", "doi", "arxiv_id", "license", "peer_review_status",
    "primary_specialist", "secondary_specialists", "tier", "quant_relevance",
    "selection_status", "selection_reason",
]

DOWNLOAD_FIELDS = [
    "pdf_path", "pdf_bytes", "pdf_sha256", "pdf_page_count", "final_url",
    "content_type", "downloaded_at_utc", "download_status", "download_error",
]

LANES = {
    "A": "evolutionary_optimization",
    "B": "neural_symbolic_mathematics",
    "C": "evolutionary_portfolio_optimization",
    "D": "constrained_offline_risk_sensitive_rl",
    "E": "dependent_data_generalization",
    "F": "execution_market_making_rl",
}


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def paper(
    cid: str,
    lane: str,
    title: str,
    authors: str,
    year: int,
    venue: str,
    landing_url: str,
    pdf_url: str,
    *,
    doi: str = "",
    arxiv_id: str = "",
    license_: str,
    peer_review_status: str,
    primary: str,
    secondary: str,
    tier: str,
    relevance: str,
) -> dict[str, str]:
    return {
        "candidate_id": cid,
        "lane": LANES[lane],
        "title": title,
        "authors": authors,
        "year": str(year),
        "venue": venue,
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "license": license_,
        "peer_review_status": peer_review_status,
        "primary_specialist": primary,
        "secondary_specialists": secondary,
        "tier": tier,
        "quant_relevance": relevance,
        "selection_status": "accepted",
        "selection_reason": "Selected by the first-lane multi-agent acquisition review; open PDF and direct mathematical relevance.",
    }


PAPERS = [
    # A — evolutionary optimization foundations and theory.
    paper("A01", "A", "Differential Evolution: A Survey and Analysis", "Tarik Eltaeib; Ausif Mahmood", 2018, "Applied Sciences 8(10):1945", "https://doi.org/10.3390/app8101945", "https://mdpi-res.com/d_attachment/applsci/applsci-08-01945/article_deploy/applsci-08-01945-v2.pdf", doi="10.3390/app8101945", license_="CC BY 4.0", peer_review_status="peer-reviewed journal article", primary="18_optimization_information_geometry_specialist", secondary="12_selection_bias_validation_specialist;13_learning_theory_generalization_specialist", tier="canonical", relevance="Derivative-free global search, adaptive mutation/crossover, and hyperparameter optimization under non-convex objectives."),
    paper("A02", "A", "The CMA Evolution Strategy: A Tutorial", "Nikolaus Hansen", 2016, "arXiv", "https://arxiv.org/abs/1604.00772", "https://arxiv.org/pdf/1604.00772", arxiv_id="1604.00772", license_="arXiv open-access copy; reuse license to verify", peer_review_status="technical tutorial/preprint", primary="18_optimization_information_geometry_specialist", secondary="13_learning_theory_generalization_specialist;19_portfolio_dro_optimal_transport_specialist", tier="canonical", relevance="Information-geometric covariance adaptation for ill-conditioned, non-convex calibration and portfolio objectives."),
    paper("A03", "A", "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", "Kalyanmoy Deb; Amrit Pratap; Sameer Agarwal; T. Meyarivan", 2002, "IEEE Transactions on Evolutionary Computation 6(2)", "https://doi.org/10.1109/4235.996017", "https://dynamics.org/~altenber/UH_ICS/EC_REFS/MULTI_OBJ/DebPratapAgarwalMeyarivan.pdf", doi="10.1109/4235.996017", license_="author-hosted open-access copy; reuse license to verify", peer_review_status="peer-reviewed journal article", primary="18_optimization_information_geometry_specialist", secondary="19_portfolio_dro_optimal_transport_specialist;12_selection_bias_validation_specialist", tier="canonical", relevance="Pareto optimization for return, drawdown, turnover, capacity, and robustness trade-offs."),
    paper("A04", "A", "Natural Evolution Strategies", "Daan Wierstra; Tom Schaul; Tobias Glasmachers; Yi Sun; Jan Peters; Jürgen Schmidhuber", 2014, "Journal of Machine Learning Research 15", "https://www.jmlr.org/papers/v15/wierstra14a.html", "https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf", license_="JMLR open-access publication; reuse terms to verify", peer_review_status="peer-reviewed journal article", primary="18_optimization_information_geometry_specialist", secondary="13_learning_theory_generalization_specialist;03_stochastic_control_hjb_bsde_specialist", tier="canonical", relevance="Natural-gradient black-box optimization connecting information geometry, policy search, and robust model calibration."),
    paper("A05", "A", "A Survey on Recent Progress in the Theory of Evolutionary Algorithms for Discrete Optimization", "Benjamin Doerr; Frank Neumann", 2020, "arXiv", "https://arxiv.org/abs/2006.16709", "https://arxiv.org/pdf/2006.16709", arxiv_id="2006.16709", license_="arXiv open-access copy; reuse license to verify", peer_review_status="survey/preprint", primary="18_optimization_information_geometry_specialist", secondary="13_learning_theory_generalization_specialist;12_selection_bias_validation_specialist", tier="theory", relevance="Runtime theory, parameter control, and stochastic/dynamic optimization needed to audit evolutionary search rather than treat it as a black box."),
    paper("A06", "A", "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", "Tim Salimans; Jonathan Ho; Xi Chen; Szymon Sidor; Ilya Sutskever", 2017, "arXiv", "https://arxiv.org/abs/1703.03864", "https://arxiv.org/pdf/1703.03864", arxiv_id="1703.03864", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="18_optimization_information_geometry_specialist", secondary="03_stochastic_control_hjb_bsde_specialist;14_neural_dynamics_sequence_specialist", tier="modern", relevance="Scalable derivative-free policy optimization and a rigorous comparator for RL-based trading controls."),

    # B — symbolic discovery and neural-symbolic mathematics.
    paper("B01", "B", "Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems", "Steven L. Brunton; Joshua L. Proctor; J. Nathan Kutz", 2016, "Proceedings of the National Academy of Sciences 113(15)", "https://doi.org/10.1073/pnas.1517384113", "https://arxiv.org/pdf/1509.03580", doi="10.1073/pnas.1517384113", arxiv_id="1509.03580", license_="arXiv open-access copy; reuse license to verify", peer_review_status="peer-reviewed journal article", primary="18_optimization_information_geometry_specialist", secondary="15_neural_operator_scientific_ml_specialist;14_neural_dynamics_sequence_specialist", tier="canonical", relevance="Sparse equation discovery for interpretable latent dynamics, regime models, and nonlinear alpha diagnostics."),
    paper("B02", "B", "AI Feynman: A Physics-Inspired Method for Symbolic Regression", "Silviu-Marian Udrescu; Max Tegmark", 2020, "Science Advances 6(16)", "https://doi.org/10.1126/sciadv.aay2631", "https://arxiv.org/pdf/1905.11481", doi="10.1126/sciadv.aay2631", arxiv_id="1905.11481", license_="arXiv open-access copy; reuse license to verify", peer_review_status="peer-reviewed journal article", primary="18_optimization_information_geometry_specialist", secondary="15_neural_operator_scientific_ml_specialist;13_learning_theory_generalization_specialist", tier="canonical", relevance="Physics-guided decomposition and symbolic recovery for interpretable model structure."),
    paper("B03", "B", "Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients", "Brenden K. Petersen; Mikel Landajuela; T. Nathan Mundhenk; Claudio P. Santiago; Soo K. Kim; Joanne T. Kim", 2021, "International Conference on Learning Representations", "https://openreview.net/forum?id=m5Qsh0kBQG", "https://arxiv.org/pdf/1912.04871", arxiv_id="1912.04871", license_="arXiv open-access copy; reuse license to verify", peer_review_status="peer-reviewed conference paper", primary="18_optimization_information_geometry_specialist", secondary="03_stochastic_control_hjb_bsde_specialist;13_learning_theory_generalization_specialist", tier="modern", relevance="Policy-gradient search over symbolic programs for compact, auditable quantitative relationships."),
    paper("B04", "B", "Neural Symbolic Regression that Scales", "Luca Biggio; Tommaso Bendinelli; Aurélien Lucchi; Giambattista Parascandolo", 2021, "Proceedings of Machine Learning Research 139", "https://proceedings.mlr.press/v139/biggio21a.html", "https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf", license_="CC BY 4.0", peer_review_status="accepted peer-reviewed conference paper", primary="18_optimization_information_geometry_specialist", secondary="13_learning_theory_generalization_specialist;14_neural_dynamics_sequence_specialist", tier="modern", relevance="Scalable transformer-guided symbolic regression for large candidate spaces and noisy data."),
    paper("B05", "B", "Symbolic Regression via Neural-Guided Genetic Programming Population Seeding", "T. Nathan Mundhenk; Mikel Landajuela; Ruben Glatt; Claudio P. Santiago; Daniel M. Faissol; Brenden K. Petersen", 2021, "arXiv", "https://arxiv.org/abs/2111.00053", "https://arxiv.org/pdf/2111.00053", arxiv_id="2111.00053", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="18_optimization_information_geometry_specialist", secondary="14_neural_dynamics_sequence_specialist;12_selection_bias_validation_specialist", tier="modern", relevance="Neural proposal models coupled to genetic-programming search for higher-quality symbolic hypotheses."),
    paper("B06", "B", "Contemporary Symbolic Regression Methods and their Relative Performance", "William La Cava; Patryk Orzechowski; Bogdan Burlacu; Fabrício Olivetti de França; Marco Virgolin; Ying Jin; Michael Kommenda; Jason H. Moore", 2021, "NeurIPS Datasets and Benchmarks / arXiv", "https://arxiv.org/abs/2107.14351", "https://arxiv.org/pdf/2107.14351", arxiv_id="2107.14351", license_="arXiv open-access copy; reuse license to verify", peer_review_status="peer-reviewed benchmark paper", primary="18_optimization_information_geometry_specialist", secondary="12_selection_bias_validation_specialist;13_learning_theory_generalization_specialist", tier="benchmark", relevance="Benchmark design and failure analysis to prevent symbolic-regression overclaiming and selection bias."),

    # C — robust evolutionary portfolio construction.
    paper("C01", "C", "Portfolio Optimization with an Envelope-Based Multi-Objective Evolutionary Algorithm", "Jürgen Branke; B. Scheckenbach; M. Stein; Kalyanmoy Deb; Hartmut Schmeck", 2009, "European Journal of Operational Research", "https://doi.org/10.1016/j.ejor.2008.01.054", "https://repository.ias.ac.in/9444/1/363.pdf", doi="10.1016/j.ejor.2008.01.054", license_="institutional repository open-access copy; reuse license to verify", peer_review_status="peer-reviewed journal article", primary="19_portfolio_dro_optimal_transport_specialist", secondary="18_optimization_information_geometry_specialist;12_selection_bias_validation_specialist", tier="canonical", relevance="Pareto-envelope construction for portfolio objectives and efficient-frontier uncertainty."),
    paper("C02", "C", "Time-Stamped Resampling for Robust Evolutionary Portfolio Optimization", "Sandra García; David Quintana; Inés M. Galván; Pedro Isasi", 2012, "Expert Systems with Applications", "https://e-archivo.uc3m.es/entities/publication/cc699ffc-a2d9-4169-a99a-ca6ba0ec6507", "https://e-archivo.uc3m.es/rest/api/core/bitstreams/da63bdbf-cb3d-4fdc-b088-2ee6bbf321a2/content", license_="institutional repository open-access copy; reuse license to verify", peer_review_status="peer-reviewed journal article", primary="19_portfolio_dro_optimal_transport_specialist", secondary="12_selection_bias_validation_specialist;05_time_series_dependence_specialist", tier="robust", relevance="Time-aware resampling for nonstationary portfolio fitness and validation leakage control."),
    paper("C03", "C", "Utilizing Dependence among Variables in Evolutionary Algorithms for Mixed-Integer Programming: A Case Study on Multi-Objective Constrained Portfolio Optimization", "Yi Chen; Aimin Zhou; Swagatam Das", 2021, "arXiv", "https://arxiv.org/abs/1909.08748", "https://arxiv.org/pdf/1909.08748", arxiv_id="1909.08748", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="19_portfolio_dro_optimal_transport_specialist", secondary="18_optimization_information_geometry_specialist;08_robust_statistics_extremes_specialist", tier="robust", relevance="Compressed coding for dependent selection/weight variables in large mixed-integer constrained portfolios."),
    paper("C04", "C", "Portfolio Optimization by Improved NSGA-II and SPEA2 Based on Different Risk Measures", "Massimiliano Kaucic; Mojtaba Moradi; Mohmmad Mirzazadeh", 2019, "Financial Innovation", "https://doi.org/10.1186/s40854-019-0140-6", "https://jfin-swufe.springeropen.com/counter/pdf/10.1186/s40854-019-0140-6.pdf", doi="10.1186/s40854-019-0140-6", license_="CC BY 4.0", peer_review_status="peer-reviewed journal article", primary="19_portfolio_dro_optimal_transport_specialist", secondary="18_optimization_information_geometry_specialist;08_robust_statistics_extremes_specialist", tier="modern", relevance="Comparative MOEA portfolio construction with practical constraints and frontier-quality diagnostics."),
    paper("C05", "C", "Multiobjective Portfolio Optimization via Pareto Front Evolution", "Yi Chen; Aimin Zhou", 2022, "Complex & Intelligent Systems", "https://doi.org/10.1007/s40747-022-00715-8", "https://link.springer.com/content/pdf/10.1007/s40747-022-00715-8.pdf", doi="10.1007/s40747-022-00715-8", license_="CC BY 4.0", peer_review_status="peer-reviewed journal article", primary="19_portfolio_dro_optimal_transport_specialist", secondary="18_optimization_information_geometry_specialist;12_selection_bias_validation_specialist", tier="modern", relevance="Direct evolution of portfolio Pareto fronts for many-objective construction and stability analysis."),
    paper("C06", "C", "EvoFolio: A Portfolio Optimization Method Based on Multi-Objective Evolutionary Algorithms", "Alfonso Guarino; Domenico Santoro; Luca Grilli; Rocco Zaccagnino; Mario Balbi", 2024, "Neural Computing and Applications", "https://doi.org/10.1007/s00521-024-09456-w", "https://link.springer.com/content/pdf/10.1007/s00521-024-09456-w.pdf", doi="10.1007/s00521-024-09456-w", license_="CC BY 4.0", peer_review_status="peer-reviewed journal article", primary="19_portfolio_dro_optimal_transport_specialist", secondary="18_optimization_information_geometry_specialist;13_learning_theory_generalization_specialist", tier="modern", relevance="Modern evolutionary portfolio framework coupling objectives, constraints, and out-of-sample evaluation."),

    # D — safe, constrained, offline, and risk-sensitive RL.
    paper("D01", "D", "Policy Gradient for Coherent Risk Measures", "Aviv Tamar; Yinlam Chow; Mohammad Ghavamzadeh; Shie Mannor", 2015, "NeurIPS 2015", "https://proceedings.neurips.cc/paper/2015/hash/024d7f84fff11dd7e8d9c510137a2381-Abstract.html", "https://proceedings.neurips.cc/paper/2015/file/024d7f84fff11dd7e8d9c510137a2381-Paper.pdf", license_="NeurIPS public proceedings copy; reuse terms to verify", peer_review_status="accepted peer-reviewed conference paper", primary="03_stochastic_control_hjb_bsde_specialist", secondary="08_robust_statistics_extremes_specialist;19_portfolio_dro_optimal_transport_specialist", tier="canonical", relevance="Policy gradients for coherent downside risk rather than reward-only trading objectives."),
    paper("D02", "D", "Constrained Policy Optimization", "Joshua Achiam; David Held; Aviv Tamar; Pieter Abbeel", 2017, "Proceedings of Machine Learning Research 70", "https://proceedings.mlr.press/v70/achiam17a.html", "https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf", license_="CC BY 4.0", peer_review_status="accepted peer-reviewed conference paper", primary="03_stochastic_control_hjb_bsde_specialist", secondary="13_learning_theory_generalization_specialist;12_selection_bias_validation_specialist", tier="canonical", relevance="Trust-region policy optimization with explicit cost constraints for safe trading agents."),
    paper("D03", "D", "Risk-Constrained Reinforcement Learning with Percentile Risk Criteria", "Yinlam Chow; Aviv Tamar; Shie Mannor; Marco Pavone", 2018, "Journal of Machine Learning Research 18", "https://www.jmlr.org/papers/v18/15-636.html", "https://jmlr.org/papers/volume18/15-636/15-636.pdf", license_="JMLR open-access publication; reuse terms to verify", peer_review_status="peer-reviewed journal article", primary="03_stochastic_control_hjb_bsde_specialist", secondary="08_robust_statistics_extremes_specialist;19_portfolio_dro_optimal_transport_specialist", tier="canonical", relevance="Chance and percentile constraints for drawdown, inventory, and tail-loss control."),
    paper("D04", "D", "Risk-Averse Trust Region Optimization for Reward-Volatility Reduction", "Luca Bisi; Luca Sabbioni; Edoardo Vittori; Matteo Papini; Marcello Restelli", 2020, "IJCAI 2020", "https://www.ijcai.org/Proceedings/2020/632", "https://www.ijcai.org/Proceedings/2020/0632.pdf", license_="IJCAI public proceedings copy; reuse terms to verify", peer_review_status="accepted peer-reviewed conference paper", primary="03_stochastic_control_hjb_bsde_specialist", secondary="08_robust_statistics_extremes_specialist;13_learning_theory_generalization_specialist", tier="modern", relevance="Trust-region control that penalizes performance volatility and unstable trading policies."),
    paper("D05", "D", "Conservative Offline Distributional Reinforcement Learning", "Yecheng Jason Ma; Dinesh Jayaraman; Osbert Bastani", 2021, "NeurIPS 2021", "https://proceedings.neurips.cc/paper/2021/hash/a05d886123a54de3ca4b0985b718fb9b-Abstract.html", "https://proceedings.neurips.cc/paper/2021/file/a05d886123a54de3ca4b0985b718fb9b-Paper.pdf", license_="NeurIPS public proceedings copy; reuse terms to verify", peer_review_status="accepted peer-reviewed conference paper", primary="03_stochastic_control_hjb_bsde_specialist", secondary="13_learning_theory_generalization_specialist;12_selection_bias_validation_specialist", tier="modern", relevance="Distributional uncertainty and conservative value learning under logged, support-limited market data."),
    paper("D06", "D", "Constrained Offline Policy Optimization", "Nicholas Polosky; Bruno C. da Silva; Madalina Fiterau; Jithin Jagannath", 2022, "Proceedings of Machine Learning Research 162", "https://proceedings.mlr.press/v162/polosky22a.html", "https://proceedings.mlr.press/v162/polosky22a/polosky22a.pdf", license_="CC BY 4.0", peer_review_status="accepted peer-reviewed conference paper", primary="03_stochastic_control_hjb_bsde_specialist", secondary="12_selection_bias_validation_specialist;13_learning_theory_generalization_specialist", tier="modern", relevance="Offline policy improvement with safety constraints and no live-market exploration."),

    # E — learning theory for dependence and nonstationarity.
    paper("E01", "E", "Rademacher Complexity Bounds for Non-I.I.D. Processes", "Mehryar Mohri; Afshin Rostamizadeh", 2008, "NeurIPS 2008", "https://proceedings.neurips.cc/paper/2008/hash/7eacb532570ff6858afd2723755ff790-Abstract.html", "https://proceedings.neurips.cc/paper/2008/file/7eacb532570ff6858afd2723755ff790-Paper.pdf", license_="NeurIPS public proceedings copy; reuse terms to verify", peer_review_status="accepted peer-reviewed conference paper", primary="13_learning_theory_generalization_specialist", secondary="05_time_series_dependence_specialist;12_selection_bias_validation_specialist", tier="canonical", relevance="Capacity and generalization bounds when financial observations are dependent rather than IID."),
    paper("E02", "E", "Stability Bounds for Stationary Phi-Mixing and Beta-Mixing Processes", "Mehryar Mohri; Afshin Rostamizadeh", 2010, "Journal of Machine Learning Research 11", "https://www.jmlr.org/papers/v11/mohri10a.html", "https://www.jmlr.org/papers/volume11/mohri10a/mohri10a.pdf", license_="JMLR open-access publication; reuse terms to verify", peer_review_status="peer-reviewed journal article", primary="13_learning_theory_generalization_specialist", secondary="05_time_series_dependence_specialist;11_sequential_inference_online_learning_specialist", tier="canonical", relevance="Algorithmic stability guarantees under mixing assumptions used in time-series validation."),
    paper("E03", "E", "Learning Theory and Algorithms for Forecasting Non-Stationary Time Series", "Vitaly Kuznetsov; Mehryar Mohri", 2015, "NeurIPS 2015", "https://proceedings.neurips.cc/paper/2015/hash/2ef3a4e0dffe7d7f6d9634e4f99a6a78-Abstract.html", "https://papers.neurips.cc/paper/5836-learning-theory-and-algorithms-for-forecasting-non-stationary-time-series.pdf", license_="NeurIPS public proceedings copy; reuse terms to verify", peer_review_status="accepted peer-reviewed conference paper", primary="13_learning_theory_generalization_specialist", secondary="05_time_series_dependence_specialist;11_sequential_inference_online_learning_specialist", tier="canonical", relevance="Nonstationary forecasting bounds and discrepancy-aware model selection for regime-changing markets."),
    paper("E04", "E", "Nonparametric Risk Bounds for Time-Series Forecasting", "Daniel J. McDonald; Cosma Rohilla Shalizi; Mark Schervish", 2017, "Journal of Machine Learning Research 18", "https://www.jmlr.org/papers/v18/13-336.html", "https://jmlr.org/papers/volume18/13-336/13-336.pdf", license_="JMLR open-access publication; reuse terms to verify", peer_review_status="peer-reviewed journal article", primary="13_learning_theory_generalization_specialist", secondary="05_time_series_dependence_specialist;12_selection_bias_validation_specialist", tier="theory", relevance="Finite-sample forecasting risk without forcing parametric or IID market assumptions."),
    paper("E05", "E", "Generalization Bounds of Recurrent Neural Networks for Temporal Data", "Minshuo Chen; Xingguo Li; Tuo Zhao", 2019, "arXiv", "https://arxiv.org/abs/1910.12947", "https://arxiv.org/pdf/1910.12947", arxiv_id="1910.12947", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="13_learning_theory_generalization_specialist", secondary="14_neural_dynamics_sequence_specialist;05_time_series_dependence_specialist", tier="modern", relevance="RNN capacity and sequence-length dependence for neural financial forecasting."),
    paper("E06", "E", "Deep Learning for Psi-Weakly Dependent Processes", "William Kengne; Modou Wade", 2023, "arXiv", "https://arxiv.org/abs/2302.00333", "https://arxiv.org/pdf/2302.00333", arxiv_id="2302.00333", license_="CC BY 4.0", peer_review_status="research preprint", primary="13_learning_theory_generalization_specialist", secondary="05_time_series_dependence_specialist;14_neural_dynamics_sequence_specialist", tier="modern", relevance="Deep-network risk guarantees under broad weak dependence beyond classical mixing."),

    # F — realistic execution and market-making RL.
    paper("F01", "F", "Market Making via Reinforcement Learning", "John Spooner; Martin Fearnley; Rahul Savani; Andreas Koukorinis", 2018, "arXiv", "https://arxiv.org/abs/1804.04216", "https://arxiv.org/pdf/1804.04216", arxiv_id="1804.04216", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="03_stochastic_control_hjb_bsde_specialist;06_high_frequency_econometrics_specialist", tier="canonical", relevance="Inventory-aware market-making policies with execution-risk objectives."),
    paper("F02", "F", "Double Deep Q-Learning for Optimal Execution", "Bingqian Ning; Franco Ho Ting Lin; Sebastian Jaimungal", 2018, "arXiv", "https://arxiv.org/abs/1812.06600", "https://arxiv.org/pdf/1812.06600", arxiv_id="1812.06600", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="03_stochastic_control_hjb_bsde_specialist;06_high_frequency_econometrics_specialist", tier="canonical", relevance="Discrete-action execution control with deep Q-learning and implementation shortfall objectives."),
    paper("F03", "F", "ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets", "Selim Amrouni; Aymeric Moulin; Jared Vann; Svitlana Vyetrenko; Tucker Balch; Manuela Veloso", 2021, "arXiv", "https://arxiv.org/abs/2110.14771", "https://arxiv.org/pdf/2110.14771", arxiv_id="2110.14771", license_="arXiv open-access copy; reuse license to verify", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="17_graph_network_systemic_risk_specialist;03_stochastic_control_hjb_bsde_specialist", tier="infrastructure", relevance="Multi-agent limit-order-market simulation for controlled counterfactual training and stress tests."),
    paper("F04", "F", "JAX-LOB: A GPU-Accelerated Limit Order Book Simulator to Unlock Large Scale Reinforcement Learning for Trading", "Sascha Frey; Kang Li; Peer Nagy; Silvia Sapora; Chris Lu; Stefan Zohren; Jakob Foerster; Anisoara Calinescu", 2023, "arXiv", "https://arxiv.org/abs/2308.13289", "https://arxiv.org/pdf/2308.13289", arxiv_id="2308.13289", license_="CC BY 4.0", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="06_high_frequency_econometrics_specialist;18_optimization_information_geometry_specialist", tier="infrastructure", relevance="Vectorized order-book simulation for scalable, reproducible execution-policy experiments."),
    paper("F05", "F", "Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation", "Hamza Bodor; Laurent Carlier", 2025, "arXiv", "https://arxiv.org/abs/2501.08822", "https://arxiv.org/pdf/2501.08822", arxiv_id="2501.08822", license_="CC BY 4.0", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="06_high_frequency_econometrics_specialist;14_neural_dynamics_sequence_specialist", tier="modern", relevance="Neural queue-reactive dynamics for realistic fills, state transitions, and policy evaluation."),
    paper("F06", "F", "Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution", "Tomas Espana; Yadh Hafsi; Fabrizio Lillo; Edoardo Vittori", 2025, "arXiv", "https://arxiv.org/abs/2511.15262", "https://arxiv.org/pdf/2511.15262", arxiv_id="2511.15262", license_="CC BY 4.0", peer_review_status="research preprint", primary="20_microstructure_hawkes_execution_specialist", secondary="03_stochastic_control_hjb_bsde_specialist;06_high_frequency_econometrics_specialist", tier="modern", relevance="Execution policy learning inside queue-reactive microstructure models with fill and impact realism."),
]


def main() -> None:
    if len(PAPERS) != 36:
        raise SystemExit(f"Expected 36 accepted papers, found {len(PAPERS)}")
    candidate_ids = [row["candidate_id"] for row in PAPERS]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit("Duplicate candidate IDs in acquisition list")
    pdf_urls = [row["pdf_url"] for row in PAPERS]
    if len(pdf_urls) != len(set(pdf_urls)):
        raise SystemExit("Duplicate PDF URLs in acquisition list")

    with CORPUS_INDEX.open(newline="", encoding="utf-8-sig") as handle:
        existing = {norm(row["title"]): row["document_id"] for row in csv.DictReader(handle)}
    duplicates = [(row["candidate_id"], existing[norm(row["title"])]) for row in PAPERS if norm(row["title"]) in existing]
    if duplicates:
        raise SystemExit(f"Existing-corpus title duplicates must be resolved: {duplicates}")

    output_fields = list(FIELDS)
    existing_rows: dict[str, dict[str, str]] = {}
    if OUTPUT.exists():
        with OUTPUT.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            output_fields.extend(field for field in (reader.fieldnames or []) if field not in output_fields)
            existing_rows = {row["candidate_id"]: row for row in reader}
    output_fields.extend(field for field in DOWNLOAD_FIELDS if field not in output_fields)
    for row in PAPERS:
        old = existing_rows.get(row["candidate_id"])
        if old and old.get("pdf_url") == row["pdf_url"]:
            for field in DOWNLOAD_FIELDS:
                row[field] = old.get(field, "")

    BATCH.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in output_fields} for row in PAPERS)
    print(f"Wrote {len(PAPERS)} accepted candidates to {OUTPUT}")


if __name__ == "__main__":
    main()
