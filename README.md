# ma-thesis-dh-code
## Repository Description
This repository provides code and material that is used in my master's thesis ("Quantifying Formulaic Flexibility of Middle High German Legal Texts"). It is programmatically functional and primarily serves the exploration of data. As such, it is to be read in conjunction with the study, and vice versa.

## File Description
### py/
* py/.ipynb contains notebooks used for the analysis and visualization of data. Except 0x, they are numbered according to the research questions
* py/ddp_util.py and formutis.py contains helper functions

## Setup
### Main
* Clone repo, create (interactive) Python environment (>=3.8; 3.10 recommended) under Linux or WSL
* Install additional dependencies depending on notebook (full requirements.txt is likely overkill).

### Optional
#### For transforming reference corpups
* Fully unzip REM tarball in /corpora/REM/ (or download via https://zenodo.org/record/3624693)
#### For custom language detection of historical German
* Download https://huggingface.co/ERCDiDip/langdetect into models/mom-langdetect

#### For sequence prediction
* Download https://huggingface.co/atzenhofer/distilroberta-base-mhg-charter-mlm into models/custom/distilroberta-base-mhg-charter-mlm
* Download https://huggingface.co/atzenhofer/xlm-roberta-base-mhg-charter-mlm into models/custom/xlm-roberta-base-mhg-charter-mlm

## Misc
Formatted using mostly https://github.com/psf/black and https://github.com/csurfer/blackcellmagic
