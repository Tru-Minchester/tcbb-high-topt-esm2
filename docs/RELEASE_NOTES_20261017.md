# Release notes: 20261017

This local release candidate is derived from the 20261010 snapshot without changing that earlier directory.

- Replaced hard-coded Windows drive paths in the included Python scripts with package-relative paths.
- Rewrote absolute paths in the nine JSON phase manifests as package-relative references.
- Kept frozen predictions, analysis outputs, Figure 1 replacement assets, and Figure 7 ROC/precision-recall QA outputs unchanged.
- Rebuilt `MANIFEST.csv` with relative paths and SHA-256 checksums after the path cleanup.
- Repository, DOI, and license fields remain explicit placeholders because no public identifiers were provided.

This package is ready for local review only. Do not upload or publish it until the repository URL, archive DOI, license, and institutional data-governance checks are finalized.
