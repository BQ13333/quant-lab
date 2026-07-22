# Z.ai GLM-OCR Markdown Pipeline

Date: 2026-07-09

## Objective

Convert the PDFs in `New_Pdf_folders_for_markdown/` into high-quality Markdown
using Z.ai GLM-OCR only.

This pipeline does not run local OCR and does not use MarkItDown as a fallback.
Local work is limited to inventory, duplicate detection, page-count preflight,
provider-safe transport packaging, resumable manifests, raw response storage,
final Markdown assembly, and quality checks.

## Script

```bash
python3 quant_lab/scripts/run_zai_ocr_pipeline.py --dry-run
```

Live run:

```bash
export ZAI_API_KEY="..."
python3 quant_lab/scripts/run_zai_ocr_pipeline.py \
  --run-id ZAI-OCR-CHUNKED-FULL-BOOKS-20260709 \
  --quality-profile optimal \
  --workers 4 \
  --timeout-seconds 1800 \
  --retries 5
```

Smoke test on one book:

```bash
export ZAI_API_KEY="..."
python3 quant_lab/scripts/run_zai_ocr_pipeline.py \
  --run-id ZAI-OCR-CHUNKED-SMOKE-20260709 \
  --book-contains "Gaussian Processes" \
  --limit-documents 1 \
  --quality-profile optimal \
  --workers 1
```

## Output Layout

```text
quant_lab/artifacts/zai_ocr/<RUN_ID>/
  manifest.json
  summary.md
  ocr_plan.csv
  ocr_manifest.csv
  ocr_quality.csv
  ocr_errors.csv
  book_manifest.csv
  chunk_plan.csv
  raw_responses/
  chunk_markdown/
  markdown/
  chunks_pdf/
  chunks_image/
```

## Current Folder Preflight

Dry run found:

- Documents in staging folder: 40.
- Ready non-duplicate documents: 38.
- Full-document runnable under Z.ai provider caps: 0.
- Exact duplicate pairs:
  - Gaussian Processes for Machine Learning.
  - Numerical Optimization.

## Optimal Profile

`--quality-profile optimal` is the default. It uses:

- `glm-ocr` through Z.ai Layout Parsing for extraction.
- Provider-safe PDF transport chunks by default.
- Hybrid transport splitting by default: fast pypdf in a child process first,
  robust Poppler fallback only for problematic PDFs.
- Page-image transport fallback for stubborn PDFs whose page ranges cannot be
  packaged as PDFs. The rendered images are upload containers only; OCR remains
  Z.ai `glm-ocr`.
- Short splitter timeouts by default: if pypdf or Poppler cannot package a
  range quickly, the runner falls back to page-image transport instead of
  stalling the OCR run.
- Raw Z.ai JSON and Markdown retention.
- Z.ai crop-image metadata request.
- Z.ai layout-visualization request.
- Longer timeout and retry defaults for large technical PDFs.
- MIME-aware data-URI upload transport by default.

It still does not run local OCR. Local page packaging only exists to satisfy
Z.ai provider limits.

Hard-PDF preflight:

- `Probabilistic Machine Learning` could not be reliably packaged as PDF ranges.
- The fallback successfully prepared `1097` ready request chunks:
  - `1` PDF chunk.
  - `1096` page-image chunks.
  - Max image size about `1.19MB`, safely under Z.ai's `10MB` image cap.
- A 20-chunk live validation on this hard PDF returned `20/20` successful
  chunks, produced about `178985` assembled Markdown characters, and recorded
  no OCR error rows.
- The completed PML run assembled `1097/1097` chunks into Markdown after a
  single-worker retry pass cleared five transient Z.ai `429` rate-limit chunks.
  Final output was about `2497151` Markdown characters with zero error rows.
- Six very-low-character chunks in the final quality manifest were manually
  checked and were valid section-divider or exercise pages, not OCR misses.

## Model Routing

Use `glm-ocr` for the actual Markdown extraction. It is the dedicated Z.ai
Layout Parsing OCR model and returns Markdown plus layout details for text,
formula, table, and image blocks.

Do not use a larger VLM as the primary bulk OCR engine unless Z.ai exposes it
through the Layout Parsing OCR endpoint. Larger models such as `GLM-5V-Turbo`
or `GLM-4.6V` are better used after OCR for targeted review:

- Review low-character chunks.
- Repair broken mathematical formulas.
- Inspect complex tables.
- Check theorem/proof sections where OCR quality is uncertain.
- Produce QA notes, not copied book passages.

## Quality Policy

Use `summary.md`, `ocr_manifest.csv`, and `ocr_quality.csv` before promoting
Markdown into the source-mining corpus.

Watch for:

- Z.ai provider limit errors.
- Returned page count not matching local page count.
- Empty Markdown.
- Very low characters per page.
- Missing formula/table blocks in formula-heavy books.

## Provider Limits

The Z.ai GLM-OCR guide and Layout Parsing API currently document PDF and page
limits. If the provider rejects a full PDF because of size or page count, the
runner uses provider-safe transport chunks in the default `chunked` mode. It
does not locally OCR the document.

Smoke test result: Z.ai returned error code `1214` for a 265-page PDF:
`PDF max 100 pages`. A follow-up test using `start_page_id=1` and
`end_page_id=1` on the same full PDF returned the same provider-limit error.
This means Z.ai checks total PDF pages before honoring page ranges.

Therefore, full-document mode can only process source PDFs that are already at
or below the provider page and size caps. Larger books require transport chunks
that are each within provider limits, while OCR itself can still remain fully
Z.ai.

## Multi-Agent Handoff

After OCR is complete, run specialist agents over the Markdown corpus:

- Probability and stochastic-process agent.
- Statistical inference and high-dimensional statistics agent.
- Optimization and control agent.
- Risk, tail, and extreme-value agent.
- Financial mathematics and derivatives agent.
- AI/ML, RL, and bandit agent.
- Quant strategy and portfolio-construction agent.
- Formula registry and implementation-spec agent.

These agents should work from Markdown plus raw layout JSON, writing paraphrased
formula registries, theorem maps, assumptions, implementation obligations, data
requirements, and validation tests. They should not copy long copyrighted
passages into downstream artifacts.
