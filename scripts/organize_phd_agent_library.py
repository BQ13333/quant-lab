#!/usr/bin/env python3
"""Materialize the verified Markdown corpus into the canonical 20-agent taxonomy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ASTRONOMY_OUTLIER_ID = "0802-0175-the-x-ray-infraredsubmillimetre-connection-and-the-legacy-era-of-cosmology"
GENERATOR_SENTINEL = ".generated_by_organize_phd_agent_library"


AI_EXTENSIONS: dict[str, dict[str, Any]] = {
    "measure_probability_martingale_specialist": {
        "ai_methods": ["probabilistic programming", "Bayesian neural networks", "normalizing flows", "diffusion processes", "uncertainty calibration"],
        "optimization_methods": ["stochastic approximation", "constrained gradient methods", "martingale control variates"],
        "quant_mission": "Make every learned alpha, risk process, and trading policy adapted, integrable, uncertainty-aware, and free of hidden look-ahead or arbitrage.",
    },
    "stochastic_calculus_rough_paths_specialist": {
        "ai_methods": ["signature methods", "Neural CDE", "Neural SDE", "Neural ODE", "rough-volatility networks", "path transformers"],
        "optimization_methods": ["adjoint sensitivity", "stochastic gradients", "kernel and signature truncation search"],
        "quant_mission": "Turn irregular market paths into filtration-safe learned representations and stable continuous-time volatility and hedging models.",
    },
    "stochastic_control_hjb_bsde_specialist": {
        "ai_methods": ["safe reinforcement learning", "actor-critic", "deep hedging", "deep BSDE", "model-based RL", "offline RL"],
        "optimization_methods": ["dynamic programming", "policy gradients", "evolution strategies", "constrained policy optimization"],
        "quant_mission": "Produce executable, cost-aware policies for hedging, execution, inventory, and market making with verification or bounded policy error.",
    },
    "numerical_pde_sde_monte_carlo_specialist": {
        "ai_methods": ["PINN", "deep BSDE", "neural PDE solver", "neural operator", "surrogate modeling", "automatic differentiation"],
        "optimization_methods": ["multilevel Monte Carlo", "variance reduction", "adaptive meshing", "Bayesian solver tuning"],
        "quant_mission": "Ensure learned pricing and risk solvers are faster only after convergence, Greeks, boundaries, and total numerical error are independently certified.",
    },
    "time_series_dependence_specialist": {
        "ai_methods": ["Transformers", "structured state-space models", "Mamba", "Gaussian processes", "deep state-space models", "regime representation learning"],
        "optimization_methods": ["spectral estimation", "state filtering", "change-aware hyperparameter search", "evolutionary regime selection"],
        "quant_mission": "Extract forecastable dependence and regimes without leakage, while forcing neural forecasts to beat robust linear and state-space baselines.",
    },
    "high_frequency_econometrics_specialist": {
        "ai_methods": ["neural denoising", "temporal convolution", "event Transformers", "neural point processes", "learned volatility and jump estimators"],
        "optimization_methods": ["asynchronous likelihood optimization", "sampling-frequency search", "noise-robust representation learning"],
        "quant_mission": "Separate latent price information from timestamp noise, bid-ask bounce, asynchronous trading, and microstructure artifacts before alpha formation.",
    },
    "high_dimensional_statistics_rmt_specialist": {
        "ai_methods": ["deep factor models", "sparse learning", "representation learning", "learned covariance", "graphical models", "feature selection"],
        "optimization_methods": ["RMT-guided shrinkage", "regularization paths", "genetic feature search", "stability selection"],
        "quant_mission": "Control p/n instability so neural features, factors, covariance estimates, and portfolio weights remain stable out of sample.",
    },
    "robust_statistics_extremes_specialist": {
        "ai_methods": ["robust losses", "adversarial training", "tail-conditioned diffusion", "extreme-value neural estimation", "rare-event generative models"],
        "optimization_methods": ["distributionally robust training", "importance sampling", "cross-entropy rare-event search", "stress scenario optimization"],
        "quant_mission": "Keep models useful under contamination, heavy tails, liquidity crises, tail dependence, and regime breaks rather than average conditions only.",
    },
    "bayesian_latent_variable_specialist": {
        "ai_methods": ["Bayesian neural networks", "variational inference", "MCMC", "SMC", "normalizing flows", "Gaussian processes", "deep latent-state models"],
        "optimization_methods": ["ELBO optimization", "Hamiltonian methods", "amortized inference", "Bayesian optimization"],
        "quant_mission": "Represent parameter, state, model, and regime uncertainty explicitly and propagate it into forecasts, risk limits, and position sizing.",
    },
    "causal_identification_invariance_specialist": {
        "ai_methods": ["double machine learning", "causal forests", "orthogonal neural networks", "invariant risk minimization", "causal representation learning", "causal GNN"],
        "optimization_methods": ["cross-fitting", "orthogonal-score optimization", "environment-invariance search", "sensitivity optimization"],
        "quant_mission": "Find drivers that survive interventions and regime changes instead of promoting unstable correlations or feature-importance stories.",
    },
    "sequential_inference_online_learning_specialist": {
        "ai_methods": ["online learning", "contextual bandits", "meta-learning", "adaptive conformal prediction", "Bayesian changepoints", "continual learning"],
        "optimization_methods": ["dynamic regret minimization", "adaptive mirror descent", "bandit optimization", "population-based adaptation"],
        "quant_mission": "Adapt models and allocations under nonstationarity with anytime-valid error control, explicit detection delay, and cost-aware feedback.",
    },
    "selection_bias_validation_specialist": {
        "ai_methods": ["AutoML governance", "neural architecture search auditing", "meta-labeling validation", "ensemble selection", "null-label learning tests"],
        "optimization_methods": ["nested optimization", "genetic-search trial accounting", "multiple-testing correction", "purged combinatorial validation"],
        "quant_mission": "Prevent ML, neural, genetic, and manual research search from manufacturing false alpha through leakage, selection, or hidden trial counts.",
    },
    "learning_theory_generalization_specialist": {
        "ai_methods": ["PAC-Bayes", "algorithmic stability", "neural tangent analysis", "representation complexity", "domain adaptation", "distribution-shift learning"],
        "optimization_methods": ["complexity regularization", "robust empirical risk", "implicit-bias diagnostics", "capacity-constrained search"],
        "quant_mission": "Translate neural and ML complexity into non-vacuous dependent-data generalization diagnostics and realistic effective sample requirements.",
    },
    "neural_dynamics_sequence_specialist": {
        "ai_methods": ["Neural ODE", "Neural CDE", "Neural SDE", "S4", "Mamba", "HiPPO", "Transformers", "continuous-time attention"],
        "optimization_methods": ["adjoint training", "solver-aware optimization", "long-context curriculum", "architecture and state-dimension search"],
        "quant_mission": "Model irregular, asynchronous, long-memory market streams with stable hidden dynamics and strict interpolation and filtration discipline.",
    },
    "neural_operator_scientific_ml_specialist": {
        "ai_methods": ["FNO", "PINO", "DeepONet", "physics-informed learning", "deep BSDE", "differentiable simulation", "inverse operators"],
        "optimization_methods": ["operator-loss balancing", "boundary-constrained training", "multi-fidelity optimization", "surrogate calibration"],
        "quant_mission": "Build reusable pricing, calibration, HJB, PDE, and risk operators whose economic constraints and extrapolation errors are certified.",
    },
    "generative_diffusion_schrodinger_specialist": {
        "ai_methods": ["score diffusion", "Schrodinger bridges", "normalizing flows", "flow matching", "GAN", "conditional path generation"],
        "optimization_methods": ["score matching", "transport optimization", "likelihood-free calibration", "rare-regime conditional sampling"],
        "quant_mission": "Generate path laws and stress scenarios that preserve temporal, cross-asset, tail, filtration, and no-arbitrage structure.",
    },
    "graph_network_systemic_risk_specialist": {
        "ai_methods": ["GNN", "temporal graph networks", "graph Transformers", "learned adjacency", "hypergraph learning", "counterparty embeddings"],
        "optimization_methods": ["graph structure learning", "evolutionary graph search", "degree-preserving null optimization", "contagion stress optimization"],
        "quant_mission": "Model genuine transmission and contagion across assets and counterparties while separating common-factor dependence and circular edge construction.",
    },
    "optimization_information_geometry_specialist": {
        "ai_methods": ["natural gradients", "mirror descent", "differentiable optimization", "Bayesian optimization", "symbolic regression", "surrogate-assisted search"],
        "optimization_methods": ["CMA-ES", "differential evolution", "genetic programming", "NSGA-II", "mixed-integer optimization", "nonconvex global search"],
        "quant_mission": "Select and certify the right optimizer for every model, including gradient, derivative-free, evolutionary, and multi-objective search with KKT and sensitivity checks.",
    },
    "portfolio_dro_optimal_transport_specialist": {
        "ai_methods": ["differentiable allocation", "deep portfolio networks", "reinforcement learning allocation", "graph portfolios", "learned ambiguity sets"],
        "optimization_methods": ["DRO", "optimal transport", "multi-objective evolutionary search", "NSGA-II", "risk-budget optimization", "robust genetic allocation"],
        "quant_mission": "Convert uncertain forecasts into feasible, diversified, capacity-aware portfolios that survive costs, estimation error, and distribution shift.",
    },
    "microstructure_hawkes_execution_specialist": {
        "ai_methods": ["neural Hawkes", "deep order-book models", "LOB Transformers", "queue representation learning", "RL execution", "learned market impact"],
        "optimization_methods": ["point-process likelihood", "constrained RL", "evolutionary calibration", "execution schedule optimization", "fill-policy search"],
        "quant_mission": "Turn signals into executable orders using realistic queues, latency, fills, impact, venue rules, and transaction costs.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def split_agents(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,|]", value or "") if part.strip()]


def metadata_by_path(workspace: Path) -> dict[Path, dict[str, str]]:
    rows = read_csv(workspace / "PhD_Math_Mega_Corpus" / "manifest.csv") + read_csv(
        workspace / "Quant_Math_PhD_Papers" / "manifest.csv"
    )
    return {(workspace / row["pdf_path"]).resolve(): row for row in rows}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def specialist_profile(
    number: int,
    specialist: dict[str, Any],
    extension: dict[str, Any],
    primary_count: int,
    secondary_count: int,
) -> str:
    methods = ", ".join(extension["ai_methods"])
    optimization = ", ".join(extension["optimization_methods"])
    obligations = "\n".join(f"- {value}" for value in specialist["proof_obligations"])
    kills = "\n".join(f"- {value}" for value in specialist["kill_criteria"])
    return f"""# {number:02d}. {specialist['title']}

Agent ID: `{specialist['id']}`
Lane: `{specialist['lane']}`
Active primary papers: {primary_count}
Secondary-routed papers: {secondary_count}

## Mathematical ownership

{specialist['mandate']}

## AI and ML depth

This specialist must be able to design, challenge, and optimize models using: {methods}.

Optimization stack: {optimization}.

Quant-fund mission: {extension['quant_mission']}

AI does not override mathematics. Every learned model must map assumptions to diagnostics, beat leakage-safe baselines, expose trial counts, quantify uncertainty, and survive regime, cost, and capacity stress.

## Proof obligations

{obligations}

## Kill criteria

{kills}

## Retrieval

- `primary_index.csv`: papers owned by this specialist.
- `secondary_index.csv`: papers owned elsewhere but routed here for cross-specialist review.
- `papers/`: one canonical Markdown file per primary paper; secondary papers are referenced, not duplicated.
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="PhD_Math_Agent_Library")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = (workspace / args.output_dir).resolve()

    specialists_config = json.loads((workspace / "quant_lab/configs/phd_math_specialists.json").read_text(encoding="utf-8"))
    specialists = specialists_config["specialists"]
    if len(specialists) != 20:
        raise SystemExit(f"Expected exactly 20 specialists, found {len(specialists)}")
    specialist_by_id = {row["id"]: row for row in specialists}
    if set(specialist_by_id) != set(AI_EXTENSIONS):
        raise SystemExit("AI-extension IDs do not exactly match the canonical 20-agent roster")

    queue = read_csv(workspace / "PhD_Math_Mega_Corpus/markdown_conversion_queue.csv")
    plan = read_csv(run_dir / "ocr_plan.csv")
    books = {row["document_id"]: row for row in read_csv(run_dir / "book_manifest_verified.csv")}
    plan_by_path = {Path(row["input_path"]).resolve(): row for row in plan}
    queue_by_path = {(workspace / row["pdf_path"]).resolve(): row for row in queue}
    metadata = metadata_by_path(workspace)
    if len(queue_by_path) != 279 or set(queue_by_path) != set(plan_by_path):
        raise SystemExit("Queue/production-plan source paths are not an exact 279-document match")

    if output.exists():
        if not (output / GENERATOR_SENTINEL).exists():
            raise SystemExit(f"Refusing to replace non-generated output directory: {output}")
        shutil.rmtree(output)
    (output / "specialists").mkdir(parents=True)
    (output / "indexes").mkdir(parents=True)
    (output / "quarantine" / "papers").mkdir(parents=True)
    (output / GENERATOR_SENTINEL).write_text("generated\n", encoding="utf-8")

    folders = {
        specialist["id"]: output / "specialists" / f"{index:02d}_{specialist['id']}"
        for index, specialist in enumerate(specialists, 1)
    }
    for folder in folders.values():
        (folder / "papers").mkdir(parents=True)

    corpus_rows: list[dict[str, Any]] = []
    primary_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    secondary_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlaps: Counter[tuple[str, str]] = Counter()
    for source_path, plan_row in sorted(plan_by_path.items(), key=lambda item: item[1]["document_id"]):
        queue_row = queue_by_path[source_path]
        document_id = plan_row["document_id"]
        primary = queue_row["primary_agent"]
        secondary = split_agents(queue_row.get("secondary_agents", ""))
        if primary not in specialist_by_id or any(agent not in specialist_by_id for agent in secondary):
            raise SystemExit(f"Unknown specialist routing for {document_id}")
        book = books[document_id]
        markdown_source = Path(book["verified_markdown_path"]).resolve()
        if not markdown_source.is_file():
            raise SystemExit(f"Verified Markdown missing: {document_id}")
        front = parse_frontmatter(markdown_source.read_text(encoding="utf-8"))
        if front.get("sha256") != plan_row["input_sha256"]:
            raise SystemExit(f"Markdown/source hash mismatch: {document_id}")
        meta = metadata.get(source_path, {})
        quarantined = document_id == ASTRONOMY_OUTLIER_ID
        if quarantined:
            target = output / "quarantine" / "papers" / f"{document_id}.md"
            retrieval_status = "excluded_non_quant_astronomy"
        else:
            target = folders[primary] / "papers" / f"{document_id}.md"
            retrieval_status = "active"
        os.link(markdown_source, target)
        relative_target = target.relative_to(output)
        title = meta.get("title") or queue_row["title"]
        authors = meta.get("authors", "")
        year = meta.get("year") or str(meta.get("published", ""))[:4]
        source_url = meta.get("landing_url") or meta.get("source_url") or meta.get("pdf_url", "")
        row = {
            "document_id": document_id,
            "title": title,
            "authors": authors,
            "year": year,
            "primary_specialist": primary,
            "secondary_specialists": ";".join(secondary),
            "tier": queue_row.get("tier", ""),
            "source": meta.get("source") or queue_row.get("source", ""),
            "source_url": source_url,
            "license": meta.get("license", ""),
            "peer_review_status": meta.get("peer_review_status", ""),
            "relevance": meta.get("relevance", ""),
            "page_count": plan_row["page_count"],
            "source_pdf_sha256": plan_row["input_sha256"],
            "original_pdf_path": str(source_path),
            "markdown_path": str(target),
            "markdown_relative_path": str(relative_target),
            "markdown_sha256": sha256_file(target),
            "verification_status": book["status"],
            "warning_pages": book.get("warning_pages", ""),
            "retrieval_status": retrieval_status,
            "exclusion_reason": "Astronomy/cosmology ingestion error; unrelated to quantitative mathematics" if quarantined else "",
        }
        corpus_rows.append(row)
        if not quarantined:
            primary_rows[primary].append(row)
            for agent in secondary:
                secondary_rows[agent].append(row)
                overlaps[(primary, agent)] += 1

    fields = list(corpus_rows[0])
    write_csv(output / "indexes/corpus_index.csv", fields, corpus_rows)
    for index, specialist in enumerate(specialists, 1):
        agent = specialist["id"]
        folder = folders[agent]
        write_csv(folder / "primary_index.csv", fields, primary_rows[agent])
        write_csv(folder / "secondary_index.csv", fields, secondary_rows[agent])
        (folder / "SPECIALIST_PROFILE.md").write_text(
            specialist_profile(index, specialist, AI_EXTENSIONS[agent], len(primary_rows[agent]), len(secondary_rows[agent])),
            encoding="utf-8",
        )
        links = [
            f"- [{row['title']}](papers/{row['document_id']}.md) — {row['year'] or 'year unknown'}"
            for row in primary_rows[agent]
        ]
        (folder / "INDEX.md").write_text(
            f"# {specialist['title']} — Primary Papers\n\n" + "\n".join(links) + "\n",
            encoding="utf-8",
        )

    roster_fields = ["number", "agent_id", "title", "lane", "primary_papers", "secondary_routes", "ai_methods", "optimization_methods", "quant_mission"]
    roster_rows = []
    for index, specialist in enumerate(specialists, 1):
        ext = AI_EXTENSIONS[specialist["id"]]
        roster_rows.append(
            {
                "number": index,
                "agent_id": specialist["id"],
                "title": specialist["title"],
                "lane": specialist["lane"],
                "primary_papers": len(primary_rows[specialist["id"]]),
                "secondary_routes": len(secondary_rows[specialist["id"]]),
                "ai_methods": ";".join(ext["ai_methods"]),
                "optimization_methods": ";".join(ext["optimization_methods"]),
                "quant_mission": ext["quant_mission"],
            }
        )
    write_csv(output / "indexes/agent_roster.csv", roster_fields, roster_rows)
    (output / "indexes/agent_capabilities.json").write_text(
        json.dumps(AI_EXTENSIONS, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matrix_fields = ["primary_specialist"] + [specialist["id"] for specialist in specialists]
    matrix_rows = []
    for primary in [specialist["id"] for specialist in specialists]:
        matrix_rows.append(
            {"primary_specialist": primary, **{secondary: overlaps[(primary, secondary)] for secondary in matrix_fields[1:]}}
        )
    write_csv(output / "indexes/secondary_overlap_matrix.csv", matrix_fields, matrix_rows)

    (output / "README.md").write_text(
        "# PhD Mathematics Agent Library\n\n"
        "This is the curated retrieval view of the verified 279-document OCR corpus. It contains exactly 20 canonical PhD-level specialist roles. "
        "Each active paper exists once under its primary specialist; secondary expertise is represented through indexes so retrieval does not overweight duplicated papers.\n\n"
        f"- Active quant/mathematics papers: {sum(len(rows) for rows in primary_rows.values())}\n"
        f"- Quarantined non-quant papers: {sum(row['retrieval_status'] != 'active' for row in corpus_rows)}\n"
        f"- Primary specialists: {len(specialists)}\n"
        f"- Secondary routing relationships: {sum(len(rows) for rows in secondary_rows.values())}\n"
        "- Formula bodies remain byte-identical to the verified OCR artifacts; categorized files are hard links.\n\n"
        "Start with `indexes/agent_roster.csv`, then open a specialist's `SPECIALIST_PROFILE.md` and `INDEX.md`.\n",
        encoding="utf-8",
    )
    (output / "ACQUISITION_GAPS.md").write_text(
        "# Priority Acquisition Gaps\n\n"
        "The current 279-document source set does not justify claiming deep corpus coverage of genetic or evolutionary optimization. "
        "Agent 18 owns the acquisition lane, challenged by Agents 12, 13, and 19.\n\n"
        "Acquire PhD-grade material on CMA-ES, differential evolution, genetic programming, NSGA-II and multi-objective search, neuroevolution, "
        "symbolic regression, robust portfolio evolutionary optimization, constrained/risk-sensitive reinforcement learning, and leakage-safe governance of adaptive search.\n",
        encoding="utf-8",
    )
    quarantine_row = next(row for row in corpus_rows if row["retrieval_status"] != "active")
    (output / "quarantine/README.md").write_text(
        "# Retrieval Quarantine\n\n"
        f"Excluded `{quarantine_row['document_id']}`: {quarantine_row['title']}. "
        "The source is astronomy/cosmology and was incorrectly described as multilevel Monte Carlo mathematics. "
        "It is retained only for provenance and is not routed to any specialist.\n",
        encoding="utf-8",
    )
    print(
        f"Organized {len(corpus_rows)} Markdown files: "
        f"{sum(len(rows) for rows in primary_rows.values())} active, 1 quarantined, "
        f"{sum(len(rows) for rows in secondary_rows.values())} secondary routes."
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
