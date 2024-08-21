#  IMPACT OF SKIN TONE DIVERSITY ON OUT-OF-DISTRIBUTION DETECTION METHODS IN DERMATOLOGY

Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin tones, thereby providing competitive quality of care across different segments of the population. Although bias and fairness assessment in skin lesion classification has been an active research area, there is substantially less exploration of the implications of skin tone representations and Out-of-Distribution (OOD) detectors' performance.  
Current OOD methods detect samples from different hardware devices, clinical settings, or unknown disease samples. However, the absence of robustness analysis across skin tones questions whether these methods are fair detectors. 
As most skin datasets are reported to suffer from bias in skin tone distribution, this could lead to higher false positive rates in a particular skin tone.  In this paper, we present a framework to evaluate OOD detectors across different skin tones and scenarios.
We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from dermatoscopic and clinical protocols. 
Our experiments yield that in poorly performing OOD models, the representation gap measured between skin types is wider (from $\approx 10\%$ to $30\%$) up for samples from darker skin tones. Compared to better performing models, skin type performance only differs for $\approx 2\%$. Furthermore, this work shows that understanding  OOD methods' performance beyond average metrics is critical to developing more fair approaches. As we observe, models with a similar overall performance have a significant difference in the representation gap, impacting FST I-IV and FST V-VI differently.

![miccai-overvew-pdf-2 pdf](https://github.com/user-attachments/assets/5c9a9dd1-c205-46a7-b581-12f71e7f631d)

## Repo organization

- See [this folder](link) for code regarding the different OOD methods trained on different proportions . We adopt Isolation Forest and OneClassSVM as baselines and AutoEncoder as state-of-the-art OOD methods. All models, training, and testing procedures, as well as hyperparameters, can be replicated following the code in this folder.

- See [this folder](link) for the datasets used to validate the proposed framework: Fitzpatrick17k, SCIN, and SD-198 for  clinical samples from different collection protocols. We stratify the samples from both datasets based on skin tones (FST I-IV and FST V-VI). The labels assigned for each dataset can be found in this folder.

- See [this notebook ](link) For the evaluation and ranking of all OOD methods studied and our $RG$ metric and  comparison.
  
- See [this notebook ](link) For the fairness metrics comparison for all OOD methods studied.

## Citation

If you find useful this repo, please consider citing:
> TBD

## Setup 

`$ python -m venv oodenv`

`$ source oodenv/bin/activate`

`$ pip install -r requirements.txt`

`$ ipython kernel install --user --name=oodenv`

`$ python -m notebook`
