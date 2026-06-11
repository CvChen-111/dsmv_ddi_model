# DSMV-DDI

A Dual-level Pharmacological Semantic and Stereochemical Visual Representation Learning Framework for Drug-Drug Interaction Prediction

## Overview

Drug-drug interactions (DDIs) are a major challenge in medication safety, particularly under polypharmacy settings. Although numerous computational methods have been developed for DDI prediction, existing approaches still face limitations in jointly modeling multi-granularity pharmacological semantics and stereochemical molecular characteristics, especially under cold-start scenarios involving previously unseen drugs. To address these limitations, we propose DSMV-DDI, a multimodal representation learning framework for drug-drug interaction prediction that integrates biomedical knowledge graph topology, chemical substructure features, dual-level pharmacological semantic representations, and stereochemical molecular visual representations derived from three-dimensional molecular conformations. In particular, the proposed dual-level semantic strategy jointly characterizes interaction-level pharmacological associations and intrinsic single-drug functional properties, enabling complementary modeling of pharmacological information across different semantic granularities. Furthermore, molecular visual representation learning captures geometric and spatial characteristics beyond topology-based molecular representations, improving generalization to topologically unseen drugs. Extensive experiments on real-world DDI datasets demonstrate that DSMV-DDI consistently outperforms multiple state-of-the-art methods, achieving an accuracy of 0.967 and an AUPR of 0.992 under the conventional setting, and further improving accuracy by 22.2% and 40.3% over the strongest baseline under partial and complete cold-start conditions, respectively. Ablation analyses further confirm that dual-level pharmacological semantics contribute most to overall classification performance, while stereochemical molecular visual representations drive the largest gains in cold-start generalization, together validating the complementary value of the proposed multimodal design.

## Installation

```bash
git clone https://github.com/CvChen-111/dsmv_ddi_model.git
cd dsmv_ddi_model
pip install torch torchvision pandas numpy scikit-learn rdkit pillow transformers

## Usage

```bash
python task1.py      # Conventional setting
python task2.py      # Partial cold-start
python task3.py      # Complete cold-start

## Citation

```bibtex
@article{chen2025dsmv,
  title={DSMV-DDI: A Dual-level Pharmacological Semantic and Stereochemical Visual Representation Learning Framework for Drug-Drug Interaction Prediction},
  author={Chen, Fangni and Jiang, Tingting and Yang, Shuai and Yuan, Xiaohui and Wang, Chao and Gu, Lichuan},
  journal={Information Fusion},
  year={2025}
}
```
