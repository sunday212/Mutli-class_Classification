Title: Imbalanced Multi-Class Classification on Fetal Cardiotocograms

Models: Principal Component Analysis and XGBoost

Author: Lisa Mok

Project Description: Classify fetal state, using XGBoost, with respect to diagnostic features using neural networks and principle component analysis to reduce dimensionality. SMOTE used for oversampling minority classes, Tomek Links used for undersamping majority classes and weighted cost-minimisation was applied.

Project Details: 
2126 foetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and the fetal state (Normal, Suspect, Pathologic). In this project we predict the classification of the  3-class foetal state.

### Attribute Information:
- LB - FHR baseline (beats per minute)
- AC - # of accelerations per second 
- FM - # of feotal movements per second 
- UC - # of uterine contractions per second
- DL - # of light decelerations per second
- DS - # of severe decelerations per second
- DP - # of prolonged decelerations per second
- ASTV - percentage of time with abnormal short term variability
- MSTV - mean value of short term variability
- ALTV - percentage of time with abnormal long term variability
- MLTV - mean value of long term variability
- Width - width of FHR histogram
- Min - minimum of FHR histogram
- Max - Maximum of FHR histogram
- Nmax - # of histogram peaks
- Nzeros - # of histogram zeros
- Mode - histogram mode
- Mean - histogram mean
- Median - histogram median
- Variance - histogram variance
- Tendency - histogram tendency
- CLASS - FHR pattern class code (1 to 10)
- NSP - fetal state class code (N=normal; S=suspect; P=pathologic)

Language: python

Dataset: cardiotoc.csv

Datasource: https://www.openml.org/d/1466 

Scripts: 
- cardio_exploration.py
- cardio_principal_component_analysis.py
- cardio_evaluation.py
- cardio_final_model.py

Data exploratory charts:
- Correlation_Matrix_of_Features.png
- Principal_Component_Analysis.png
- Target Class Distribution.png
- Histogram___.png

Tuning charts: 
- Epochs___.png
- Subsampling___.png
- Tuning_no_trees_vs_learning_rate.png
- Tuning_no_trees_vs_max_layers.png

Final model charts:
- Feature_Importance.png
- Tree.png


Acknowledgements: Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318, [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)
