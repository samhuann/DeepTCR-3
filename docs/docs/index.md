# DeepTCR3

## Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data

DeepTCR3 is a python package that has a collection of unsupervised and supervised 
deep learning methods to parse TCRSeq data. To see examples of how the algorithms can 
be used on an example datasets, see the subdirectory 'tutorials' for a collection of tutorial 
use cases across multiple datasets. For complete documentation for all available methods,
 click [here](https://sidhomj.github.io/DeepTCR3/).

While DeepTCR3 will run with Tensorflow-CPU versions, for optimal training times, 
we suggest training these algorithms on GPU's (requiring CUDA, cuDNN, and tensorflow-GPU). 

DeepTCR3 now has the added functionality of being able to analyze paired alpha/beta chain inputs as well
as also being able to take in v/d/j gene usage and the contextual HLA information the TCR-Sequences
were seen in (i.e. HLA alleles for a repertoire from a given human sample). For detailed instructions on 
how to upload this type of data, refer to the documentation for loading data into DeepTCR3.  

For questions or help, email: johnwilliamsidhom@gmail.com

## Publication

For full description of algorithm and methods behind DeepTCR3, refer to the following manuscript:

[Sidhom, J. W., Larman, H. B., Pardoll, D. M., & Baras, A. S. (2021). DeepTCR3 is a deep learning framework for revealing sequence concepts within T-cell repertoires. Nat Commun 12, 1605](https://www.nature.com/articles/s41467-021-21879-w)

## Dependencies

See requirements.txt for all DeepTCR3 dependencies. Installing DeepTCR3 from Github repository or PyPi will install all required dependencies.
It is recommended to create a virtualenv and installing DeepTCR3 within this environment to ensure proper versioning of dependencies.

In the most recent release (DeepTCR3 2.0, fifth release), the package now uses python 3.7 & Tensorflow 2.0. Since this has required an overhaul in a lot of the code, there could be some bugs so we would greatly appreciate if you post any issues to the issues page and I will do my best to fix them as quickly as possible. One can find the latest DeepTCR3 1.x version under the v1 branch if you still want to use that version. Or one can specifically pip install the specific version desired.

Instructions on how to create a virtual environment can be found here:
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Installation

In order to install DeepTCR3:

```python
pip3 install DeepTCR3

```

Or to install latest updated versions from Github repo:
 
Either download package, unzip, and run setup script:

```python
python3 setup.py install
```

Or use:

```python
pip3 install git+https://github.com/sidhomj/DeepTCR3.git

```
