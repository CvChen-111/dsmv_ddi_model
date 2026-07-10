# DSMV-DDI

**DSMV-DDI: A Dual-level Pharmacological Semantic and Stereochemical Visual Representation Learning Framework for Drug-Drug Interaction Prediction**

DSMV-DDI is a multimodal representation learning framework for multi-class drug-drug interaction (DDI) prediction. It integrates multi-source knowledge graph structural representations, chemical substructure representations, dual-level pharmacological semantic representations, and molecular visual representations derived from 3D molecular conformations.

## Overview

Drug-drug interactions are an important medication safety issue, especially in polypharmacy scenarios. DSMV-DDI is designed to predict DDI event types by jointly modeling complementary drug information from four modalities:

- **M1: Multi-source knowledge graph structural encoding**  
- **M2: Chemical substructure encoding**  
- **M3: Dual-level pharmacological semantic encoding**  
- **M4: Molecular visual encoding**  

The code supports three experimental settings:

- **Task 1:** Predict interactions between known drugs.
- **Task 2:** Predict interactions between known drugs and novel drugs.
- **Task 3:** Predict interactions between novel drugs.

## Project Structure

```text
dsmv_ddi_code/
|-- code/
|   |-- task1.py           
|   |-- task2.py               
|   |-- task3.py            
|   |-- modeltask1.py     
|   |-- modeltask2.py         
|   |-- modeltask3.py  
|   |-- se/           
|   |-- cse/         
|   |-- des/               
|   `-- img/                 
|-- dataset/
|   |-- dataset1.txt         
|   |-- dataset2.txt            
|   |-- dataset3.txt       
|   |-- dataset4.txt         
|   `-- event.db               
|-- result/                      
|-- README.md
`-- requirements.txt
```


## Dataset files

```text
dataset/dataset1.txt
dataset/dataset2.txt
dataset/dataset3.txt
dataset/dataset4.txt
dataset/event.db
```

If `dataset/event.db` is provided as `event.rar`, please extract it first and place `event.db` directly under the `dataset/` directory.

## Installation

We recommend using a Conda environment with Python 3.8 or a compatible version.

```bash
conda create -n dsmv-ddi python=3.8
conda activate dsmv-ddi
pip install -r requirements.txt
```

The main dependencies include:

```text
torch
torchvision
torch-geometric
numpy
pandas
scikit-learn
rdkit
transformers
pyarrow
pillow
tqdm
matplotlib
```

Please install the PyTorch and PyTorch Geometric versions that match your CUDA version. If CUDA is unavailable, the scripts can also run on CPU, but training will be slower.

## Usage

All training scripts should be executed from the `code/` directory because the project uses relative paths to load datasets and feature files.

```bash
cd code
```

### Task 1: Conventional DDI Prediction

Task 1 evaluates DDI prediction between known drugs using five-fold cross-validation.

```bash
python task1.py
```

Optional example:

```bash
python task1.py --epoches 120 --batch_size 1024 --seed 0
```

### Task 2: Partial Cold-start DDI Prediction

Task 2 evaluates DDI prediction between known drugs and novel drugs.

```bash
python task2.py
```

Optional example:

```bash
python task2.py --epoches 100 --batch_size 1024 --seed 1
```

### Task 3: Complete Cold-start DDI Prediction

Task 3 evaluates DDI prediction between novel drugs.

```bash
python task3.py
```

Optional example:

```bash
python task3.py --epoches 120 --batch_size 1024 --seed 1
```

## Output Files

After running the scripts, the evaluation results are saved as CSV files.

For Task 1 and Task 2, outputs are saved under:

```text
result/
```

Typical files include:

```text
alltask1.csv
eachtask1.csv
alltask2.csv
eachtask2.csv
```

For Task 3, the current script writes results to:

```text
code/result/
```

Typical files include:

```text
code/result/alltask3.csv
code/result/eachtask3.csv
```

## Main Evaluation Metrics

The scripts report multiple evaluation metrics, including:

- Accuracy
- AUPR
- AUC
- F1-score
- Precision
- Recall

For Task 1, both micro and macro metrics are calculated. For the cold-start tasks, macro F1, macro precision, and macro recall are used to better reflect performance on diverse DDI event classes.

## Reproducing the Main Experiments

To reproduce the three main experimental settings, run:

```bash
cd code
python task1.py
python task2.py
python task3.py
```

Please ensure that all required feature files are placed in the expected directories before running the scripts.

## Notes

- The scripts use fixed random seeds by default, but minor numerical differences may occur across hardware, CUDA versions, and package versions.
- The project uses relative file paths. Running the scripts from outside the `code/` directory may cause file loading errors.
- Large pretrained feature files and molecular image files should be placed in the exact paths listed above.

## Citation

If you use this code or dataset, please cite the corresponding DSMV-DDI manuscript.
