# SOmics reviewer-revision reproducibility materials

These are **cleaned archival copies**, not independently validated runnable notebooks. The original code cells and their order are retained, except for broken attempts to write/import `somics_spatial_inference.py`, unrelated directory search commands, and exploratory inference cells that emitted errors. Notebook outputs were cleared. Original notebooks should be preserved separately.

## Files
- `notebooks/SOmics_spatial_pseudosamples_cleaned.ipynb`: original training/spatial analysis, pseudo sample generation (including final 160 pseudo samples), and plotting.
- `notebooks/SOmics_reviewer_analyses_cleaned.ipynb`: original training/spatial analysis, TCGA bootstrap confidence intervals, tissue level bootstrap, leave one out, and normalization sensitivity.

## Important limitations and setup still required
1. These notebooks use absolute Colab Drive paths such as `/content/drive/MyDrive/ML_ExternalVal_FINAL/` and separate TCGA/Visium input folders. Replace or configure these paths to point to your own downloaded data.
2. They depend on original files and variables created in previous cells. Execute in order only after supplying input datasets and dependencies. No independent clean run test has been completed.
3. The full original Visium and TCGA datasets and their download/access instructions, software versions, and R/Seurat processing code should be documented in the repository's main README. A `requirements.txt` is already present in the main GitHub repository but its contents must be checked against the actual execution environment.
4. The saved `somics_spatial_inference.py` script is *not included* in these cleaned notebooks. Upload and document the correct independently maintained version if it is necessary to reproduce application inference.
5. Machine learning artifacts (RF/LR saved models, ordered 1,000 gene feature list) should be checked against notebook outputs and the published app.
6. Pseudo samples are resamples of spatial spots **within eight original tissue sections**, not 160 independent tissues/patients. Tissue level intervals are exploratory (four sections per tumor type).
7. TCGA bootstrap CIs are based on 5,000 stratified resamples of the held-out predictions; models were not refitted for each iteration.
8. Natural log versus log2 CPM sensitivity does **not** harmonize FPKM and CPM measurement differences.

## Outputs to add after checking
- `TCGA_Bootstrap_95CI.csv` (reportedly saved under `My Drive/ML Train Data/`)
- `Tissue_Level_Bootstrap_95CI.csv`
- `LOSO_RF_LR_fracCAF.csv`
- `PseudoMore/PseudoMore_PerSample_summary_fixedRealQ75.csv`
- `PseudoMore/PseudoMore_manifest_fixedRealQ75.csv`
- `PseudoMore/FixedRealQ75_thresholds.csv`
- `Normalization_Sensitivity/Normalization_PerSample_Comparison.csv`
- `Normalization_Sensitivity/Sparsity_vs_CAF_Probability.csv`
- final two-panel Figure 16 (original data-based plotted panels)

**Before release:** confirm reproducible steps with a fresh environment, remove sensitive output/credentials, check runtime package versions, and pin the tested commit/release. Only then state that the main results can be reproduced end to end.
