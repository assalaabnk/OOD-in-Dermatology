#  IMPACT OF SKIN TONE DIVERSITY ON OUT-OF-DISTRIBUTION DETECTION METHODS IN DERMATOLOGY

Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin
tones, thereby providing competitive quality of care across different population segments. Although bias and fairness assessment in skin lesion classification has been an active research area, substantially less exploration has been done of the implications of skin tone representations on Out-of-Distribution (OOD) detectors’ performance. Most skin datasets are reported to suffer from bias in skin tone distribution, which could lead to skewed model performance across skin tones. This paper explores the impact of variations of representation rates across skin tones during the training of OOD detectors and their downstream implications on performance. We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from different clinical protocols. Our experiments conducted using multiple skin image datasets reveal that increasing the representation of FST V-VI during training reduces the representation gap by ≈ 5%. We also observe an increase in the overall performance metrics for FST V-VI when more representation is shown during training. Furthermore, the group fairness metrics evaluation yields that increasing the FST V-VI representation leads to improved group fairness.

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
