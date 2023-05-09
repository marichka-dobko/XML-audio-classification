# Interpretable Audio Classification


In this work we explore the interpretable methods for audio classification. 
We show how an encoder-decoder architecture can be used for delivering meaningful 
visual explanations with the use of the prototyping technique. We focus on training and evaluating an explainable model 
called APNet proposed by Zinemanas et al., for various audio classification tasks, including identifying musical instruments, urban sounds, and respiratory 
sounds captured during a medical examination. We analyze the visual explanations provided by the trained APNet and examine the patterns 
found within these model interpretations. We show the model's ability to generalize to unseen data and the challenges of adapting it to a 
new task. 


This repository is based on original APNet implementation - https://github.com/pzinemanas/APNet.
For instance,`notebooks` folder was unchanged from the authors' repo.


## Data
We train and evaluate the APNet model on 
1. Medley-solos-DB
2. IRMAS
3. UrbanSound8k
4. ICBHI Respiratory Sound Database

To download IRMAS dataset, please refer to the source - https://www.upf.edu/web/mtg/irmas. 

The data for ICBHI can be found at the challenge [website](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).

The dataloaders for all these datasets including custom classes for IRMAS and ICBHI, can be found in `apnet/datasets.py`


## Running scripts
To train the model:
```
cd experiments
python train.py -m APNet -d MedleySolosDb -f MelSpectrogram -fold train
```

