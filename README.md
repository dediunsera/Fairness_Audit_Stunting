# Fairness Audit on Machine Learning Models for Childhood Stunting Prediction

This repository contains the experimental framework, auditing pipelines, and evaluation metrics for assessing **algorithmic fairness and demographic parity** in machine learning models used for childhood stunting prediction.

## 📌 Research Context
Machine learning models deployed for public health screening and stunting risk assessment must be equitable across diverse demographic groups. Disparities in model performance—such as varying false positive rates (FPR) or false negative rates (FNR) across gender, socioeconomic quintiles, geographic regions, and urban/rural settlements—can lead to biased healthcare resource allocation.

This project implements comprehensive **fairness audits** to quantify, evaluate, and mitigate algorithmic biases in predictive models for childhood stunting.

## 🎯 Objectives
- **Subgroup Performance Evaluation:** Measure predictive performance across protected and vulnerable subgroups (e.g., socioeconomic status, gender, region).
- **Fairness Metrics Auditing:** Compute standard fairness metrics including Equalized Odds, Demographic Parity, Predictive Parity, and Disparate Impact.
- **Bias Mitigation Strategies:** Explore pre-processing, in-processing, and post-processing approaches to ensure equitable risk estimation without significantly compromising overall predictive accuracy.

## 🔒 Data Privacy
Survey and clinical health data used in this study are confidential:
- Raw datasets (`data/`, `.xlsx`, `.csv`) are strictly excluded via `.gitignore` to protect sensitive information.
- Only reproducible scripts, anonymized aggregated metrics, and evaluation plots are versioned.

## 📂 Repository Structure
*(Structure will be populated as experiments are added)*
- `scripts/`: Source code for model training, fairness evaluations, and auditing pipelines.
- `results/`: Aggregated performance metrics, fairness disparity tables, and logs.
- `figures/`: Disparity charts, ROC curves, and subgroup fairness trade-off visualizations.
