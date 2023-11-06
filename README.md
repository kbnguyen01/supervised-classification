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

Put all the data, the 5353_util.py, and supervised_classifications.py in the same folder and run the supervised_classifications.py

Here are the results:
1) Breast cancer dataset:
{'accuracy': 0.9166666666666666, 'B': {'precision': 0.926440329218107, 'recall': 0.9166666666666666, 'f1': 0.9190863862505655}, 'M': {'precision': 0.926440329218107, 'recall': 0.9166666666666666, 'f1': 0.9190863862505655}}
{'accuracy': 0.9736842105263158, 'B': {'precision': 0.9750671684727815, 'recall': 0.9736842105263158, 'f1': 0.9739919975377039}, 'M': {'precision': 0.9750671684727815, 'recall': 0.9736842105263158, 'f1': 0.9739919975377039}}
{'accuracy': 0.9517543859649122, 'B': {'precision': 0.9542867585630744, 'recall': 0.9517543859649122, 'f1': 0.9524494968135627}, 'M': {'precision': 0.9542867585630744, 'recall': 0.9517543859649122, 'f1': 0.9524494968135627}}
{'accuracy': 0.9736842105263158, 'B': {'precision': 0.9762726488352028, 'recall': 0.9736842105263158, 'f1': 0.9741325931461514}, 'M': {'precision': 0.9762726488352028, 'recall': 0.9736842105263158, 'f1': 0.9741325931461514}}
=> Support Vector Classification is the best model for this dataset according to the result

2) Hyperthyroidism dataset:
{'accuracy': 0.9781312127236581, 'secondary toxic': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'T3 toxic': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'hyperthyroid': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'goitre': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'negative': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}}
{'accuracy': 0.9781312127236581, 'secondary toxic': {'precision': 0.9672954853008408, 'recall': 0.9781312127236581, 'f1': 0.9718295980971603}, 'T3 toxic': {'precision': 0.9672954853008408, 'recall': 0.9781312127236581, 'f1': 0.9718295980971603}, 'hyperthyroid': {'precision': 0.9672954853008408, 'recall': 0.9781312127236581, 'f1': 0.9718295980971603}, 'goitre': {'precision': 0.9672954853008408, 'recall': 0.9781312127236581, 'f1': 0.9718295980971603}, 'negative': {'precision': 0.9672954853008408, 'recall': 0.9781312127236581, 'f1': 0.9718295980971603}}
{'accuracy': 0.3856858846918489, 'secondary toxic': {'precision': 0.9559898310923588, 'recall': 0.3856858846918489, 'f1': 0.5311246270260211}, 'T3 toxic': {'precision': 0.9559898310923588, 'recall': 0.3856858846918489, 'f1': 0.5311246270260211}, 'hyperthyroid': {'precision': 0.9559898310923588, 'recall': 0.3856858846918489, 'f1': 0.5311246270260211}, 'goitre': {'precision': 0.9559898310923588, 'recall': 0.3856858846918489, 'f1': 0.5311246270260211}, 'negative': {'precision': 0.9559898310923588, 'recall': 0.3856858846918489, 'f1': 0.5311246270260211}}
{'accuracy': 0.9821073558648111, 'secondary toxic': {'precision': 0.9773506955551147, 'recall': 0.9821073558648111, 'f1': 0.9796194056205302}, 'T3 toxic': {'precision': 0.9773506955551147, 'recall': 0.9821073558648111, 'f1': 0.9796194056205302}, 'hyperthyroid': {'precision': 0.9773506955551147, 'recall': 0.9821073558648111, 'f1': 0.9796194056205302}, 'goitre': {'precision': 0.9773506955551147, 'recall': 0.9821073558648111, 'f1': 0.9796194056205302}, 'negative': {'precision': 0.9773506955551147, 'recall': 0.9821073558648111, 'f1': 0.9796194056205302}}
=> Support Vector Classification is the best model for this dataset according to the result

3) Cervical_cancer dataset:
{'accuracy': 0.9069767441860465, '0': {'precision': 0.8802185486130568, 'recall': 0.9069767441860465, 'f1': 0.892328910448471}, '1': {'precision': 0.8802185486130568, 'recall': 0.9069767441860465, 'f1': 0.892328910448471}}
{'accuracy': 0.9302325581395349, '0': {'precision': 0.8653326122228232, 'recall': 0.9302325581395349, 'f1': 0.8966096945923226}, '1': {'precision': 0.8653326122228232, 'recall': 0.9302325581395349, 'f1': 0.8966096945923226}}
{'accuracy': 0.0872093023255814, '0': {'precision': 0.8187292358803987, 'recall': 0.0872093023255814, 'f1': 0.04861977689544337}, '1': {'precision': 0.8187292358803987, 'recall': 0.0872093023255814, 'f1': 0.04861977689544337}}
{'accuracy': 0.9302325581395349, '0': {'precision': 0.8653326122228232, 'recall': 0.9302325581395349, 'f1': 0.8966096945923226}, '1': {'precision': 0.8653326122228232, 'recall': 0.9302325581395349, 'f1': 0.8966096945923226}}
=> Support Vector Classification and K-nearest Neighbors Classifier are the best models for this dataset

4) Liver_cancer dataset:
{'accuracy': 0.35435435435435436, '2': {'precision': 0.8034121077599339, 'recall': 0.35435435435435436, 'f1': 0.34217445021766746}, '1': {'precision': 0.8034121077599339, 'recall': 0.35435435435435436, 'f1': 0.34217445021766746}}
{'accuracy': 0.5525525525525525, '2': {'precision': 0.8218084705772805, 'recall': 0.5525525525525525, 'f1': 0.588057420772634}, '1': {'precision': 0.8218084705772805, 'recall': 0.5525525525525525, 'f1': 0.588057420772634}}
{'accuracy': 0.9159159159159159, '2': {'precision': 0.9132416169029441, 'recall': 0.9159159159159159, 'f1': 0.9133011120869708}, '1': {'precision': 0.9132416169029441, 'recall': 0.9159159159159159, 'f1': 0.9133011120869708}}
{'accuracy': 0.2012012012012012, '2': {'precision': 0.040481923364806244, 'recall': 0.2012012012012012, 'f1': 0.06740240240240239}, '1': {'precision': 0.040481923364806244, 'recall': 0.2012012012012012, 'f1': 0.06740240240240239}}
=> Gaussian Naive Bayes and XGBoost, are combined in a voting ensemble, is the best model for this dataset
