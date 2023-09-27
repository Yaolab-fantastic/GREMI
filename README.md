# GRAMI-Net

By capturing complementary information from multiple omics data, multi-omics integration has demonstrated promising performance in disease prediction. As the number of omics data increases, effectively representing the data and avoiding mutual interference becomes challenging due to the intricate relationships within and among omics data. Here, we propose a novel multi-omics integration framework that improves diagnostic prediction. Specifically, our framework involves constructing co-expression and co-methylation networks for each subject, followed by applying multi-level graph attention to incorporate biomolecule interaction information. The true-class-probability strategy is employed to evaluate omics-level confidence for classification, and the loss is designed using an adaptive mechanism to leverage both within- and across-omics information. Extensive experiments demonstrate that the proposed framework outperforms state-of-the-art methods on classification tasks, and indicate that the integration of three omics yields superior performance compared to employing only one or two data types. 

## Requirment

- Python 3.6
- Pytorch 1.10.2
- pytorch geometric
- sklearn
- numpy

# Usage
The data used can be obtained through https://github.com/txWang/MOGONET.
We also provide the in-house processed data of extra three diseases.
In our study, data from three omics were merged into one file. 
You can get the classification result by running model-test.py.

# Disclaimer

This tool is for research purpose and not approved for clinical use.

# Coypright

This tool is developed in Yao Lab.

The copyright holder for this project is Yao Lab.

All rights reserved.
