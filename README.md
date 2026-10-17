# SOmics

SOmics is a spatial transcriptomics and machine learning platform for analyzing immune and stromal patterns in ovarian cancer.

## SOmics GUI

The interactive SOmics web application can be accessed here:

[Launch the SOmics GUI] https://somics.streamlit.app/
For the proper function of the user interface, upload a zip file for the barcodes, feature, and an mtz of zip file for matrix files. Then upload tissue position in csv format. Finally, tissue image and scale factor. This will make the GUI function properly and produce proper results. There is also an option for a pre-converted spatial omics files, upload expression in csv format, tissue position file in csv format, and finally tissue image and scale factor will lead to proper function. An example dataset it found in a folder called "EXAMPLE DATASET SOMICS". 
## Installation and Setup

SOmics is a Python-based application developed using Streamlit. The following steps explain how to install and run the application locally.

### 1. Download the repository

```bash
git clone https://github.com/sivaGU/SOmics.git
cd SOmics
```

### 2. Create a Python environment

We recommend using Python 3.13 for the application environment.

```bash
python3.13 -m venv somics_env
```

Activate the environment:

Mac/Linux:
```bash
source somics_env/bin/activate
```

Windows:
```bash
somics_env\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch SOmics

```bash
streamlit run Somics.py
```

The application should open in a web browser. Users can upload 10x Genomics Visium data or preprocessed CSV files and select either the Random Forest or logistic regression model.

### 5. Reproducing the manuscript analyses

The machine learning analyses supporting the manuscript will be provided in the `reviewer_revision_analyses` folder.

These notebooks include model evaluation, bootstrap confidence intervals, pseudo-sample robustness testing, and normalization sensitivity analysis. Researchers should obtain the original datasets from the sources listed in the manuscript and configure the input paths before running the notebooks.
