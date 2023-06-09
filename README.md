# Metody_Selekcji

This repository contains the code used for our experiments with selection methods in the context of Computational Intelligence course at AGH UST.

Here, you will find several custom selection methods developed by us, along with the complete setup for testing them against classical methods on various optimisation problems.

Features:
- save data to file so that it can be reused for plotting later (comment out `run_selection` in `main.py` to open data from existing files)
- plot found minimums by simulation progress, standard deviations of specimens and how the diversity of solutions impact the results
- easily extendable for future selections

How to run:
1. Make sure you have a recent version of python installed. I'm using `3.11.3`
2. `pip install -r requirements.txt`
3. `python main.py`
