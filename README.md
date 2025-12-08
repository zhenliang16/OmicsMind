# OmicsMind: A Transformer Based Multi-omics Data Imputation Tool

## Requirements
OmicsMind requires Python 3.8+ and standard machine learning libraries such as PyTorch and NumPy.
Python 12 is recommended.

## Installation Steps:
1. Clone or download this project.
2. Inside the OmicsMind directory, run:
```pip install -e .```

## Running the Examples:
To try out the provided multi-omics imputation examples:
1. Download the example datasets from: https://drive.google.com/drive/folders/1njo2BXWDHMgchrAByPcFdTSnEW-bkzCW?usp=drive_link
2. Place the downloaded files into: 
```Examples/data```
3. Open Jupyter Lab or Jupyter Notebook and run one of the example notebooks:
``` Examples/run_OOL.ipynb ```
or 
``` Examples/run_GBM.ipynb ``` 
4. Chose python3 kernel.

## Project Description
OmicsMind is a generative AI–powered imputation toolkit designed to recover missing values in multi-omics datasets.
It combines a Variational Autoencoder (VAE) with a Transformer module to:
* Encode heterogeneous omics inputs into a shared latent space
* Learn cross-omics feature relationships
* Decode back into biologically meaningful scales

## License
This project is released under the MIT License.