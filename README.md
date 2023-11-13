# OOD-in-Dermatology
Identifying out of distribution samples and study  its fairness  in Dermatology

## Abstract
Addressing representation issues in dermatological settings is crucial due to variations in how skin conditions manifest across skin tones, thereby providing competitive quality of care across different segments of the population. Although bias and fairness assessment in skin lesion classification has been an active research area, there is substantially less exploration of the implications of skin tone representations and Out-of-Distribution (OOD) detectors' performance.  
Current OOD methods detect samples from different hardware devices, clinical settings, or unknown disease samples. However, the absence of robustness analysis across skin tones questions whether these methods are fair detectors. 
As most skin datasets are reported to suffer from bias in skin tone distribution, this could lead to higher false positive rates in a particular skin tone.  In this paper, we present a framework to evaluate OOD detectors across different skin tones and scenarios.
We review and compare state-of-the-art OOD detectors across two categories of skin tones, FST I-IV (lighter tones) and FST V-VI (brown and darker tones), over samples collected from dermatoscopic and clinical protocols. 
Our experiments yield that in poorly performing OOD models, the representation gap measured between skin types is wider (from $\approx 10\%$ to $30\%$) up for samples from darker skin tones. Compared to better performing models, skin type performance only differs for $\approx 2\%$. Furthermore, this work shows that understanding  OOD methods' performance beyond average metrics is critical to developing more fair approaches. As we observe, models with a similar overall performance have a significant difference in the representation gap, impacting FST I-IV and FST V-VI differently.


## Code: 
This directory includes the implementation of various ML algorithms and OOD detection approaches. We will explore different architectures and techniques to enhance the robustness of the dermatology diagnosis model.

![approach ISIB drawio](https://github.com/assalaabnk/OOD-in-Dermatology/assets/61749380/ca19aa3c-3365-4063-b0cd-3644d405922e)


## Datasets: 
We provide the Fitzpatrick 17k dataset consisting of 16,577 clinical images of 114 different skin conditions, annotated with Fitzpatrick skin type labels Skin, the images were labeled as FST I-IV and FST V-VI . Additionally, we included the ISIC 2019 dataset consisting of 25,331  images available for the classification of dermoscopic images among nine different diagnostic categories: Melanoma. Melanocytic nevus. Basal cell carcinoma.


## Contribution
