# Wave 5 HMM and Regime-Inference Research Benchmark

Version: `2026-07-15`

## Purpose

Measure whether a research scout can recover authoritative, complementary, and adversarial sources for
the library's HMM/regime gaps without hallucinating identifiers, review status, or reuse rights.

## Gap tasks

1. Identifiability and maximum-likelihood consistency.
2. Unknown state count, order estimation, and sticky persistence.
3. Explicit duration and nonparametric semi-Markov models.
4. Switching linear/nonlinear state-space systems.
5. Covariate-dependent transition probabilities.
6. Recursive online EM and particle learning.
7. Variational particle inference.
8. Neural structured inference and identifiable nonlinear states.
9. Limits and failure modes of amortized inference.
10. Robustness to contamination, misspecification, and heavy tails.
11. Filtering-safe financial allocation with turnover, costs, and execution delay.
12. Simpler changepoint and non-HMM challengers.

## Evaluation splits

Development prompts may expose gap descriptions and target counts, never target titles or identifiers.
Four later holdout prompts must be frozen separately and withheld from the scout. This first deterministic
pilot evaluates the full curated manifest; it is not a claim of hidden-holdout model performance.

## Evidence doctrine

- Identity keys: DOI, arXiv ID, then exact normalized title.
- Every proposed result must have an opened authoritative landing page.
- Peer-review status and open-licence status use enums and require evidence URLs when asserted.
- `unknown` or `review_required` is acceptable; a falsely permissive assertion is an automatic failure.
- Search snippets, third-party summaries, and citation graphs can generate leads but cannot support admission.
- Watchlist papers remain outside acquisition and OCR until explicitly admitted.

## Non-goals

The benchmark does not assess theorem correctness, empirical reproducibility, market capacity, PDF-to-
Markdown quality, or commercial training rights. Those are separate downstream gates.
