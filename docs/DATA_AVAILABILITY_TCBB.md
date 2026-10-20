# Data and Code Availability Statement

This release candidate contains the processed data, frozen model-derived files, analysis outputs, figure source files, and code needed to reproduce the reported high-Topt classification analyses. It is prepared for deposition at a public code repository and a durable data archive, but it has not yet been uploaded.

The package includes the strict no-mixed internal benchmark, the strict Seq2Topt-derived novel40 external validation set, the Seq2Topt no-exact-overlap external validation set, the public Topt no-exact-overlap validation set, AAC/DPC and ESM-2 feature matrices, frozen per-sample predictions, bootstrap and calibration outputs, screening and EC-stratified analyses, and the source/QA files for Figures 1-7.

Raw BRENDA and other third-party source database dumps are not redistributed where their original terms restrict redistribution. Processed benchmark records, provenance fields, sequence identifiers, sequences, labels, feature tables, prediction outputs, result summaries, and executable scripts are included instead.

The intended deposition split is:

- GitHub: source code, small result summaries, requirements, figure-generation scripts, and reproduction instructions.
- Zenodo or IEEE DataPort: the complete processed package, large feature matrices, per-sample predictions, figures, and `MANIFEST.csv`.

The following fields must be completed at public deposition and must not be guessed:

- Repository URL: `[GitHub URL]`
- Archived dataset identifier: `[Zenodo or IEEE DataPort DOI]`
- License: record the selected license after checking restrictions on third-party source data.

The local package version is `20261010`; the June package `ESM2_high_topt_public_release_20260603` remains unchanged as an earlier provenance snapshot.
