# GREMI: an Explainable Multi-omics Integration Framework for Enhanced Disease Prediction and Module Identification

Multi-omics integration has demonstrated promising performance in complex disease prediction. However, existing research typically focuses on maximizing prediction accuracy, while often neglecting the essential task of discovering meaningful biomarkers. This issue is particularly important in biomedicine, as molecules often interact rather than function individually to influence disease outcomes. To this end, we propose a two-phase framework named GREMI to assist multi-omics classification and explanation. In the prediction phase, we propose to improve prediction performance by employing a graph attention architecture on sample-wise co-functional networks to incorporate biomolecular interaction information for enhanced feature representation, followed by the integration of a joint-late mixed strategy and the true-class-probability block to adaptively evaluate classification confidence at both feature and omics levels. In the interpretation phase, we propose a multi-view approach to explain disease outcomes from the interaction module perspective, providing a more intuitive understanding and biomedical rationale. We incorporate Monte Carlo tree search (MCTS) to explore local-view subgraphs and pinpoint modules that highly contribute to disease characterization from the global-view. Extensive experiments demonstrate that the proposed framework outperforms state-of-the-art methods in seven different classification tasks, and our model effectively addresses data mutual interference when the number of omics types increases. We further illustrate the functional- and disease-relevance of the identified modules, as well as validate the classification performance of discovered modules using an independent cohort.

# Overview

![Framework](framework.png)

Our framework involves:
- Constructing co-expression and co-methylation networks for each subject
- Applying multi-level graph attention to incorporate biomolecule interaction information
- Utilizing a true-class-probability strategy to evaluate omics-level confidence for classification
- Designing the loss using an adaptive mechanism to leverage both within- and across-omics information

Extensive experiments demonstrate that the proposed framework outperforms state-of-the-art methods on classification tasks and indicates that the integration of three omics yields superior performance compared to employing only one or two data types.

# Requirements

- Python 3.6
- PyTorch 1.10.2
- PyTorch Geometric
- scikit-learn
- numpy

Create a conda environment using the provided `environment.yml` file:

```sh
conda env create -f environment.yml
conda activate greml-env
 ``` 

# Data Preparation
The data used can be obtained through https://github.com/txWang/MOGONET. We also provide the in-house processed data of extra three diseases. In our study, data from three omics were merged into one file. 

# Usage
## Setting up the environment

1. Clone the repository and navigate to the project directory:
```
git clone https://github.com/Yaolab-fantastic/GREMI.git
cd GREMI
```
2. Activate the conda environment:
```
conda activate greml-env
```
3. Ensure you are in the correct working directory:
```
cd /path/to/your/project
```

## Running the model
To get the classification result, run the following command:
```
python model-test.py
```
## Results
After running the model, you can expect to see the classification results which demonstrate the efficacy of our multi-omics integration approach.

## Explain

We provide a concise demo implementation for subgraph explanation using the MUTAG dataset.
The demo script (subgraph_demo.py) loads the MUTAG dataset, trains a graph neural network, and generates subgraph explanations. The output will include visualizations and metrics to help understand the model's decision-making process.
To run the demo, follow these steps:
```
cd explain
python subgraph_demo.py
```

# Disclaimer

This tool is for research purpose and not approved for clinical use. The demo showcases how to generate and interpret subgraph explanations within graph neural networks (GNNs).

# Coypright

This tool is developed in Yao Lab.

The copyright holder for this project is Yao Lab.

All rights reserved.

# Acknowledgment
This work was inspired by Alicja Chaszczewicz, Kyle Swanson, and Mert Yuksekgonul as part of the Stanford CS224W course project.

# Citation
If you use this framework in your research, please cite our work:

 ``` 
@article {Liang2023.03.19.533326,
    author = {Liang, Hong and Luo, Haoran and Sang, Zhiling and Jia, Miao and Jiang, Xiaohan and Wang, Zheng and Yao, Xiaohui and Cong, Shan},
    title = {GREMI: an Explainable Multi-omics Integration Framework for Enhanced Disease Prediction and Module Identification},
    elocation-id = {2023.03.19.533326},
    year = {2023},
    doi = {10.1101/2023.03.19.533326},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2023/11/15/2023.03.19.533326},
    eprint = {https://www.biorxiv.org/content/early/2023/11/15/2023.03.19.533326.full.pdf},
    journal = {bioRxiv}
}
 ``` 
