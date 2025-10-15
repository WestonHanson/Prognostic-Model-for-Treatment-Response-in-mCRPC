# &nbsp;&nbsp;Prognostic Model for Treatment Response in mCRPC <img src="misc/graph.png" width="140" align="left">

&nbsp;&nbsp;An XGBoost machine learning algorithm built to determine treatment response in mCRPC ultra-low pass whole genome sequencing (ULP-WGS) data.

<br></br>

## Description
This script uses a machine learning algorithm called XGBoost, which is a gradient boosting algorithm, to predict response groups. It uses genomic data such as RNA-seq data or TFBS data from tools like *[Triton](https://github.com/GavinHaLab/TritonNP)*, and uses features such as tumor fraction (TFx), fraction genome altered (FGA), loss of heterozygozity, and tumor mutational burden as covariates.

## Inputs
All inputs can be changed in main.py under the *USER IMPUT* section.

## Outputs
A model will be saved in `/saved-models` 