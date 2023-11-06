# supervised-classification
Apply supervised classification algorithms using biomedical data (specifically structured data). The goal is to identify the best supervised classification algorithm for each dataset

Dataset 1: Breast Cancer. This dataset provides imaging results (30 features) along with a gold
standard breast cancer diagnosis (malignant vs. benign) for 569 patients. The dataset description can be
found at the links below. The data file itself is named wdbc.data.
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

Dataset 2: Hyperthyroidism. This dataset provides attributes for the detection of hyperthyroidism
and related conditions (e.g., T3 toxic, goiter/goitre). The dataset has 3592 patients with 29 features, many
of which are missing values. There are two data files, allhyper.data and allhyper.test,
corresponding to the original train/test split. But ignore their split and combine them into a single set.
https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.names

Dataset 3: Cervical Cancer. This dataset provides 28 risk factor-related features for cervical cancer
for 858 patients. Note that there are several potential prediction variables of interest.
We will use Biopsy. You should not use Dx:Cancer, Dx:CIN, Dx:HPV, Dx, Hinselmann, Schiller, or
Citology as input features. The data can be found in the risk_factors_cancer.csv file.
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

Dataset 4: Liver Cancer. This dataset provides 10 features (mostly liver function test results) related
to liver cancer for 583 patients. As with many of the other datasets in this project, the data is not well-
balanced. The data file is named Indian Liver Patient Dataset (ILPD).csv.
https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29
All four of these datasets are supervised classification with a small number of classes (three of the dataset
are binary, the hyperthyroidism dataset has 5 classes). 

