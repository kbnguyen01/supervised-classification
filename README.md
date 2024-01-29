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
We will use Biopsy. The data can be found in the risk_factors_cancer.csv file.
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
{'accuracy': 0.9210526315789473, 'M': {'precision': 0.9318551082443292, 'recall': 0.9210526315789473, 'f1': 0.9235170728430973}, 'B': {'precision': 0.9318551082443292, 'recall': 0.9210526315789473, 'f1': 0.9235170728430973}}
{'accuracy': 0.9824561403508771, 'M': {'precision': 0.9836455545643771, 'recall': 0.9824561403508771, 'f1': 0.9826613316918026}, 'B': {'precision': 0.9836455545643771, 'recall': 0.9824561403508771, 'f1': 0.9826613316918026}}
{'accuracy': 0.9517543859649122, 'M': {'precision': 0.9542867585630744, 'recall': 0.9517543859649122, 'f1': 0.9524494968135627}, 'B': {'precision': 0.9542867585630744, 'recall': 0.9517543859649122, 'f1': 0.9524494968135627}}
{'accuracy': 0.9736842105263158, 'M': {'precision': 0.9762726488352028, 'recall': 0.9736842105263158, 'f1': 0.9741325931461514}, 'B': {'precision': 0.9762726488352028, 'recall': 0.9736842105263158, 'f1': 0.9741325931461514}} 
=> K-nearest neighbors is the best model for this dataset according to the result

2) Hyperthyroidism dataset:
{'accuracy': 0.9781312127236581, 'goitre': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'secondary toxic': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'hyperthyroid': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'negative': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}, 'T3 toxic': {'precision': 0.9751544153880138, 'recall': 0.9781312127236581, 'f1': 0.9752194029339667}}
{'accuracy': 0.9774685222001326, 'goitre': {'precision': 0.9664781265458889, 'recall': 0.9774685222001326, 'f1': 0.9709348769234546}, 'secondary toxic': {'precision': 0.9664781265458889, 'recall': 0.9774685222001326, 'f1': 0.9709348769234546}, 'hyperthyroid': {'precision': 0.9664781265458889, 'recall': 0.9774685222001326, 'f1': 0.9709348769234546}, 'negative': {'precision': 0.9664781265458889, 'recall': 0.9774685222001326, 'f1': 0.9709348769234546}, 'T3 toxic': {'precision': 0.9664781265458889, 'recall': 0.9774685222001326, 'f1': 0.9709348769234546}}
{'accuracy': 0.38966202783300197, 'goitre': {'precision': 0.9561637973534661, 'recall': 0.38966202783300197, 'f1': 0.5353131142317378}, 'secondary toxic': {'precision': 0.9561637973534661, 'recall': 0.38966202783300197, 'f1': 0.5353131142317378}, 'hyperthyroid': {'precision': 0.9561637973534661, 'recall': 0.38966202783300197, 'f1': 0.5353131142317378}, 'negative': {'precision': 0.9561637973534661, 'recall': 0.38966202783300197, 'f1': 0.5353131142317378}, 'T3 toxic': {'precision': 0.9561637973534661, 'recall': 0.38966202783300197, 'f1': 0.5353131142317378}}
{'accuracy': 0.9834327369118622, 'goitre': {'precision': 0.9785047624840806, 'recall': 0.9834327369118622, 'f1': 0.9809195086820885}, 'secondary toxic': {'precision': 0.9785047624840806, 'recall': 0.9834327369118622, 'f1': 0.9809195086820885}, 'hyperthyroid': {'precision': 0.9785047624840806, 'recall': 0.9834327369118622, 'f1': 0.9809195086820885}, 'negative': {'precision': 0.9785047624840806, 'recall': 0.9834327369118622, 'f1': 0.9809195086820885}, 'T3 toxic': {'precision': 0.9785047624840806, 'recall': 0.9834327369118622, 'f1': 0.9809195086820885}}
=> Support Vector Classification is the best model for this dataset according to the result

3) Cervical_cancer dataset:
{'accuracy': 0.9563953488372093, '1': {'precision': 0.9633558549381754, 'recall': 0.9563953488372093, 'f1': 0.9589290903079064}, '0': {'precision': 0.9633558549381754, 'recall': 0.9563953488372093, 'f1': 0.9589290903079064}}
{'accuracy': 0.936046511627907, '1': {'precision': 0.9251025991792066, 'recall': 0.936046511627907, 'f1': 0.9141749723145072}, '0': {'precision': 0.9251025991792066, 'recall': 0.936046511627907, 'f1': 0.9141749723145072}}
{'accuracy': 0.8197674418604651, '1': {'precision': 0.9418496404030258, 'recall': 0.8197674418604651, 'f1': 0.8600956891625162}, '0': {'precision': 0.9418496404030258, 'recall': 0.8197674418604651, 'f1': 0.8600956891625162}}
{'accuracy': 0.9593023255813954, '1': {'precision': 0.9648940897644792, 'recall': 0.9593023255813954, 'f1': 0.9613707317470798}, '0': {'precision': 0.9648940897644792, 'recall': 0.9593023255813954, 'f1': 0.9613707317470798}}
=> Support Vector Classification and Decision Tree are the best models for this dataset

4) Liver_cancer dataset:
{'accuracy': 0.35435435435435436, '2': {'precision': 0.8034121077599339, 'recall': 0.35435435435435436, 'f1': 0.34217445021766746}, '1': {'precision': 0.8034121077599339, 'recall': 0.35435435435435436, 'f1': 0.34217445021766746}}
{'accuracy': 0.5525525525525525, '2': {'precision': 0.8218084705772805, 'recall': 0.5525525525525525, 'f1': 0.588057420772634}, '1': {'precision': 0.8218084705772805, 'recall': 0.5525525525525525, 'f1': 0.588057420772634}}
{'accuracy': 0.9159159159159159, '2': {'precision': 0.9132416169029441, 'recall': 0.9159159159159159, 'f1': 0.9133011120869708}, '1': {'precision': 0.9132416169029441, 'recall': 0.9159159159159159, 'f1': 0.9133011120869708}}
{'accuracy': 0.2012012012012012, '2': {'precision': 0.040481923364806244, 'recall': 0.2012012012012012, 'f1': 0.06740240240240239}, '1': {'precision': 0.040481923364806244, 'recall': 0.2012012012012012, 'f1': 0.06740240240240239}}
=> Gaussian Naive Bayes and XGBoost, are combined in a voting ensemble, is the best model for this dataset
