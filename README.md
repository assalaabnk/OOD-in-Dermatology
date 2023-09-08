# OOD-in-Dermatology
Identifying out of distribution samples and study  its fairness  in Dermatology

## Overview
Most machine learning (ML) models assume ideal conditions, assuming that test data comes from the same distribution as the training samples. However, this assumption is rarely met in real-world applications, especially in clinical settings, where variations in hardware devices, environmental conditions, and patient characteristics introduce shifts in the input data distribution. In the field of dermatology, where automated medical disease diagnosis models are gaining traction, it is essential to address the robustness of these models towards such variations, specifically towards out-of-distribution (OOD) samples.

## Project Goal
The main goal of this project is to explore multiple ML solutions that effectively detect OOD samples before making any diagnostic decision. By identifying OOD samples, we aim to enhance the reliability and accuracy of medical disease diagnosis models in diverse clinical settings, where unknown or unfamiliar conditions may arise. Additionally, we will evaluate the fairness of these OOD detectors across different skin tones to ensure equitable performance for all patient groups.

## Repository Contents
This GitHub repository contains the code, datasets, and documentation for the project. The main components of the repository are as follows:

## Code: 
This directory includes the implementation of various ML algorithms and OOD detection approaches. We will explore different architectures and techniques to enhance the robustness of the dermatology diagnosis model.

## Datasets: 
We provide the Fitzpatrick 17k dataset consisting of 16,577 clinical images of 114 different skin conditions, annotated with Fitzpatrick skin type labels. Additionally, we will include datasets for OOD samples to test the performance of the OOD detection methods.

## Documentation: 

## Contribution