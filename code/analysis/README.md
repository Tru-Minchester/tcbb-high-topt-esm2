# Dated analysis scripts

The phase 0-7 scripts resolve the package root from their own file location, so they do not depend on a developer's local drive letter. Phase outputs are stored under `results/analysis/phaseN/`; figure outputs are stored under `figures/phase7_final/`.

The scripts preserve the original analysis conventions. Re-running a phase can require optional upstream tools (for example CD-HIT, model downloads, or third-party source data) and may create new outputs; it does not alter the frozen reference files unless the user explicitly chooses to overwrite them. The published Figure 7 is generated from frozen per-sample probabilities; no model retraining or curve smoothing is part of that figure workflow.
