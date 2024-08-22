# EVALUATING THE IMPACT OF SKIN TONE REPRESENTATION ON OUT-OF-DISTRIBUTION DETECTION PERFORMANCE IN DERMATOLOGY

Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin tones, thereby providing competitive quality of care across different segments of the population. Although bias and fairness assessment in skin lesion classification has been an active research area, there is substantially less exploration of the implications of skin tone representations and Out-of-Distribution (OOD) detectors' performance.  
Current OOD methods detect samples from different hardware devices, clinical settings, or unknown disease samples. However, the absence of robustness analysis across skin tones questions whether these methods are fair detectors. 
As most skin datasets are reported to suffer from bias in skin tone distribution, this could lead to higher false positive rates in a particular skin tone.  In this paper, we present a framework to evaluate OOD detectors across different skin tones and scenarios.
We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from dermatoscopic and clinical protocols. 
Our experiments yield that in poorly performing OOD models, the representation gap measured between skin types is wider (from $\approx 10\%$ to $30\%$) up for samples from darker skin tones. Compared to better performing models, skin type performance only differs for $\approx 2\%$. Furthermore, this work shows that understanding  OOD methods' performance beyond average metrics is critical to developing more fair approaches. As we observe, models with a similar overall performance have a significant difference in the representation gap, impacting FST I-IV and FST V-VI differently.

![approach ISIB drawio](https://github.com/assalaabnk/OOD-in-Dermatology/assets/61749380/30ca973e-c55d-40c0-b57a-c5d0906a8c0d)

## Repo organization

- See [this folder](https://github.com/assalaabnk/OOD-in-Dermatology/tree/d3d60f0c0f718db7a0cc600440fd38db87c9a831/OOD%20methods) for code regarding the different OOD methods evaluated. We adopt Isolation Forest and OneClassSVM as baselines and AutoEncoder, Neural Network Softmax, and ODIN as state-of-the-art OOD methods. All models, training, and testing procedures, as well as hyperparameters, can be replicated following the code in this folder.

- See [this folder](https://github.com/assalaabnk/OOD-in-Dermatology/tree/c24a33db1b3de81a9d380e16aa10942ebf2e4545/data) for the datasets used to validate the proposed framework: ISIC 2019 and Fitzpatrick 17k for dermoscopic and clinical samples from different collection protocols. We stratify the samples from both datasets based on skin tones (FST I-IV and FST V-VI). The labels assigned for each dataset can be found in this folder.

- See [this notebook ](https://github.com/assalaabnk/OOD-in-Dermatology/blob/582777c4ebc0428c5ce0684f91a68bd21add94df/OOD%20methods/RG___Evaluation.ipynb) For the evaluation and ranking of all OOD methods studied and our $RG$ metric comparison.

## Citation

If you find this repo useful, please consider citing:

> @inproceedings{benmalek2024evaluating,
  title={Evaluating the Impact of Skin Tone Representation on Out-of-Distribution Detection Performance in Dermatology},
  author={Benmalek, Assala and Cintas, Celia and Tadesse, Girmaw Abebe and Daneshjou, Roxana and Varshney, Kush and Dalila, Cherifi},
  booktitle={IEEE International Symposium on Biomedical Imaging},
  year={2024}}

## Setup 

`$ python -m venv oodenv`

`$ source oodenv/bin/activate`

`$ pip install -r requirements.txt`

`$ ipython kernel install --user --name=oodenv`

`$ python -m notebook`
