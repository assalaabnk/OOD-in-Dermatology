# Papers in this repository

## 1. EVALUATING THE IMPACT OF SKIN TONE REPRESENTATION ON OUT-OF-DISTRIBUTION DETECTION PERFORMANCE IN DERMATOLOGY

Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin tones, thereby providing competitive quality of care across different segments of the population. Although bias and fairness assessment in skin lesion classification has been an active research area, there is substantially less exploration of the implications of skin tone representations and Out-of-Distribution (OOD) detectors' performance.  
Current OOD methods detect samples from different hardware devices, clinical settings, or unknown disease samples. However, the absence of robustness analysis across skin tones questions whether these methods are fair detectors. 
As most skin datasets are reported to suffer from bias in skin tone distribution, this could lead to higher false positive rates in a particular skin tone.  In this paper, we present a framework to evaluate OOD detectors across different skin tones and scenarios.
We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from dermatoscopic and clinical protocols. 
Our experiments yield that in poorly performing OOD models, the representation gap measured between skin types is wider (from $\approx 10\%$ to $30\%$) up for samples from darker skin tones. Compared to better performing models, skin type performance only differs for $\approx 2\%$. Furthermore, this work shows that understanding  OOD methods' performance beyond average metrics is critical to developing more fair approaches. As we observe, models with a similar overall performance have a significant difference in the representation gap, impacting FST I-IV and FST V-VI differently.

![approach ISIB drawio](https://github.com/assalaabnk/OOD-in-Dermatology/assets/61749380/30ca973e-c55d-40c0-b57a-c5d0906a8c0d)


## Citation

If you find useful this repo, please consider citing:

> @inproceedings{benmalek2024evaluating,
  title={Evaluating the Impact of Skin Tone Representation on Out-of-Distribution Detection Performance in Dermatology},
  author={Benmalek, Assala and Cintas, Celia and Tadesse, Girmaw Abebe and Daneshjou, Roxana and Varshney, Kush and Dalila, Cherifi},
  booktitle={IEEE International Symposium on Biomedical Imaging},
  year={2024}}

[![Demo](https://img.shields.io/badge/Demo-Green?style=for-the-badge&logo=appveyor)](https://github.com/assalaabnk/OOD-in-Dermatology)

## 2. IMPACT OF SKIN TONE DIVERSITY ON OUT-OF-DISTRIBUTION DETECTION METHODS IN DERMATOLOGY

Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin
tones, thereby providing competitive quality of care across different population segments. Although bias and fairness assessment in skin lesion classification has been an active research area, substantially less exploration has been done of the implications of skin tone representations on Out-of-Distribution (OOD) detectors’ performance. Most skin datasets are reported to suffer from bias in skin tone distribution, which could lead to skewed model performance across skin tones. This paper explores the impact of variations of representation rates across skin tones during the training of OOD detectors and their downstream implications on performance. We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from different clinical protocols. Our experiments conducted using multiple skin image datasets reveal that increasing the representation of FST V-VI during training reduces the representation gap by ≈ 5%. We also observe an increase in the overall performance metrics for FST V-VI when more representation is shown during training. Furthermore, the group fairness metrics evaluation yields that increasing the FST V-VI representation leads to improved group fairness.


![miccai-overvew-pdf-2 pdf](https://github.com/user-attachments/assets/5c9a9dd1-c205-46a7-b581-12f71e7f631d)

If you find useful this repo, please consider citing:

> TBD


[![Demo](https://img.shields.io/badge/Demo-Green?style=for-the-badge&logo=appveyor)]([https://github.com/assalaabnk/OOD-in-Dermatology](https://github.com/assalaabnk/OOD-in-Dermatology/tree/OOD-detection-with-proportions))


## Setup 

`$ python -m venv oodenv`

`$ source oodenv/bin/activate`

`$ pip install -r requirements.txt`

`$ ipython kernel install --user --name=oodenv`

`$ python -m notebook`
