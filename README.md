
<h1>Prognostic Model for Treatment Response in mCRPC<img src="misc/graph.png" width="140" align="left" style="margin-right: 15px;"></h1>

An XGBoost machine learning algorithm built to determine treatment response in mCRPC ultra-low pass whole genome sequencing (ULP-WGS) data.

<br/>

## Description
This script uses a machine learning algorithm called XGBoost, which is a gradient boosting algorithm, to predict response groups. It uses genomic data such as RNA-seq data or TFBS data from tools like *[Triton](https://github.com/GavinHaLab/TritonNP)*, and uses features such as tumor fraction (TFx), fraction genome altered (FGA), loss of heterozygozity, and tumor mutational burden as covariates.

## Inputs
All inputs can be changed in main.py under the *USER IMPUT* section.

## Outputs
A model will be saved in `/saved-models` 

## Acknowledgments
This script was developed by Weston Hanson in the Gavin Ha Lab, Fred Hutchinson Cancer Center, under the supervision of Robbert D. Patton and Patrick McDeed.

## License
The MIT License (MIT)

Copyright (c) 2025 Fred Hutchinson Cancer Center

Permission is hereby granted, free of charge, to any government or not-for-profit entity, or to any person employed at one of the foregoing (each, an "Academic Licensee") who obtains a copy of this software and associated documentation files (the “Software”), to deal in the Software purely for non-commercial research and educational purposes, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or share copies of the Software, and to permit other Academic Licensees to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

No Academic Licensee shall be permitted to sell or use the Software or derivatives thereof in any service for commercial benefit. For the avoidance of doubt, any use by or transfer to a commercial entity shall be considered a commercial use and will require a separate license with Fred Hutchinson Cancer Center.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
