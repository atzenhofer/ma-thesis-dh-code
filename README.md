# ma-thesis-dh-code
## Repo Description
This repository provides code and material that is used in my Master's Thesis. It is programmatically functional and primarily serves the exploration of data.

### py
* py/.ipynb are notebooks used for the analysis and visualization of data. Except the 0s, they are numbered according to the research questions
* py/ddp_util.py and formutis.py contains helper functions

## Setup
* Create Python environment (>=3.8; 3.10 recommended) under Linux or WSL
* Install requirements.txt
* Fully unzip REM tarball in /corpora/REM/ (or download via https://zenodo.org/record/3624693)
* Download https://huggingface.co/atzenhofer/distilroberta-base-mhg-charter-mlm into models/custom/distilroberta-base-mhg-charter-mlm
* Download https://huggingface.co/atzenhofer/xlm-roberta-base-mhg-charter-mlm into models/custom/xlm-roberta-base-mhg-charter-mlm

## Misc
Formatted using mostly https://github.com/psf/black and https://github.com/csurfer/blackcellmagic
