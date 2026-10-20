# ESM-2 High-Topt Enzyme Classification

Homology-controlled, externally validated, and calibration-aware screening of high-temperature-optimum enzymes using frozen ESM-2 protein language model embeddings (ESM2-150M) and a Random Forest classifier.

Companion code repository for the manuscript submitted to *IEEE/ACM Transactions on Computational Biology and Bioinformatics* (under review).

## Repository contents

| Path | Contents |
|---|---|
| `code/` | All reproducibility scripts: pipeline steps 01–36 (data processing, feature extraction, evaluation) and `code/analysis/` phase 0–7 analyses (bootstrap CI, screening, calibration, ablation, interpretability, EC stratification, figure generation) |
| `docs/` | Data-availability statement, release notes, and analysis design notes |
| `results/` | Small result summary tables (CSV) |
| `figures/` | Submission figures and QA files for Fig. 1–7 (`phase7_final/`), plus legacy Figure 5 provenance assets (`legacy_20260603/`) |
| `MANIFEST.csv` | File-level manifest of the full release package with SHA-256 checksums (the full package also contains large feature matrices and per-sample predictions archived separately) |

## Data archive

The complete release package (247 MB, including frozen ESM-2 150M/650M feature matrices, per-sample prediction tables, and the full `MANIFEST.csv`) is archived separately:

- **Zenodo / IEEE DataPort**: *DOI to be assigned at deposition* — see `docs/DATA_AVAILABILITY_TCBB.md` for the deposition plan and third-party data terms.

After downloading, unpack the archive and place it so that `data_processed/`, `features/`, and `results/analysis/` sit next to `code/` from this repository.

## Frozen external evaluation panels

The primary model is ESM2-150M with uncalibrated probabilities. Panel identity is matched by sample count and positive count:

| Panel | N | Positives | Overlap control |
|---|---:|---:|---|
| novel40 (strict low-homology) | 66 | 26 | <40% identity to every internal sequence (CD-HIT-2D); nested within the Seq2Topt no-exact panel |
| Seq2Topt no-exact | 136 | 33 | exact-overlap removal only; superset of novel40 |
| Public Topt no-exact | 984 | 313 | exact-overlap removal; independent source (biomap-research optimal-temperature collection) |

## Reproduction

1. Install Python dependencies: see `code/requirements.txt` (pandas, NumPy, scikit-learn, imbalanced-learn, transformers, PyTorch, tqdm, requests). CD-HIT/CD-HIT-2D is required only for re-running sequence clustering from FASTA inputs.
2. Download the data archive (above) and place it next to `code/`.
3. Run the phase scripts in `code/analysis/` from any working directory; paths resolve relative to the package layout. Included outputs under `results/analysis/` (in the data archive) are the frozen reference results for comparison.
4. Compare generated outputs with the frozen references and the figure QA files under `figures/phase7_final/`.

The package does not retrain or refit the published models during figure regeneration. Raw third-party database dumps (e.g., BRENDA) are not redistributed per their licenses; processed records, sequence identifiers, and provenance fields are provided instead.

## License

Code in this repository is released under the MIT License (see `LICENSE`). Processed data in the companion archive are released under CC-BY 4.0 with source attribution; third-party source data retain their original terms (BRENDA, UniProt CC-BY 4.0, biomap-research optimal-temperature Apache-2.0, Seq2Topt as per its authors).

## Citation

If you use this code or data, please cite the manuscript (citation details to be added upon publication).
