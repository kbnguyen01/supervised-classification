import sklearn
from sklearn.impute import SimpleImputer  # use Imputer if sklearn v0.19
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import util_5353

# Read and Preprocess data
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def read_data(dataset_id):
    data = None
    data_y = None
    categorical_columns = [] 
    if dataset_id == "breast_cancer":
        data = pd.read_csv("wdbc.data", header=None)
        data = data.drop(0, axis=1)
        column_names = ['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave points1', 'symmetry1', 'fractal dimension1',
        'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave points2', 'symmetry2', 'fractal dimension2', 
        'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave points3', 'symmetry3', 'fractal dimension3']
        data.columns = column_names
        categorical_columns = ['Diagnosis']
    elif dataset_id == "hyperthyroidism":
        data_part1 = pd.read_csv("allhyper.data", header=None, delimiter=',')
        data_part2 = pd.read_csv("allhyper.test", header=None, delimiter=',')
        data_part1 = data_part1.reset_index(drop=True)
        data_part2 = data_part2.reset_index(drop=True)
        data = pd.concat([data_part1, data_part2], ignore_index=True) 
        data = data.replace('?', np.nan)
        data=data.replace({"F":1,"M":0})
        data=data.replace({"f":0,"t":1})
        column_names = [
            'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
            'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
            'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
            'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4',
            'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referall source', 'classes'
        ]
        data.columns = column_names
        encoder= LabelEncoder()
        data['referall source'] = encoder.fit_transform(data['referall source'])
        data = data.drop(columns = ['TBG'])
        data['classes'] = data['classes'].apply(lambda x: x.split('.')[0] if '.' in x else x)
        categorical_columns = [
            'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
            'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
            'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
            'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured', 'classes'
        ]
        label_columns = [
            'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
            'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
            'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
            'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']
        encoder = LabelEncoder()
        for column in label_columns:
            data[column] = encoder.fit_transform(data[column])
        categorical_columns_with_nan = [col for col in categorical_columns if data[col].isnull().any()]
        for col in categorical_columns_with_nan:
            most_frequent_value = data[col].mode().values[0]  
            data[col].fillna(most_frequent_value, inplace=True)  
    elif dataset_id == "cervical_cancer":
        data = pd.read_csv("risk_factors_cervical_cancer.csv", sep=',')
        data = data.replace('?', np.nan)
        data['Biopsy'] = data['Biopsy'].astype(str)
        data = data.drop(columns = ['Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology'])
        categorical_columns = ['Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes',
    'STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'Biopsy']
        categorical_columns_with_nan = [col for col in categorical_columns if data[col].isnull().any()]
        for col in categorical_columns_with_nan:
            most_frequent_value = data[col].mode().values[0]  
            data[col].fillna(most_frequent_value, inplace=True)
        numeric_columns = [col for col in data.columns if data[col].dtype == 'float64']
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].mean())
        label_columns = ['Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes',
    'STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B', 'STDs:HPV']
        encoder = LabelEncoder()
        for column in label_columns:
            data[column] = encoder.fit_transform(data[column])
    elif dataset_id == "liver_cancer":
        data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv", delimiter=',', header=None)
        column_names = ['age', 'gender', 'total Bilirubin', 'direct Bilirubin', 'total proteins', 'albumin', 'A/G ratio', 'SGPT', 'SGOT', 'Alkphos', 'classes'
        ]
        data.columns = column_names
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data['classes'] = data['classes'].astype(str)
        data=data.replace({"Female":1,"Male":0})
        numeric_columns = [col for col in data.columns if data[col].dtype == 'float64']
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].mean())
        encoder = LabelEncoder()
        data['gender'] = encoder.fit_transform(data['gender'])
        X = data.drop('classes', axis=1)
        y = data['classes']
        k_best = SelectKBest(score_func=f_classif, k=9)
        X_selected = k_best.fit_transform(X, y)
        selected_feature_indices = k_best.get_support(indices=True)
        selected_feature_names = X.columns[selected_feature_indices]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
        data = pd.concat([X_selected_df, y], axis=1)
        smote = SMOTE(sampling_strategy='minority', random_state=1)
        X_resampled, y_resampled = smote.fit_resample(X_selected_df, y)
        data = pd.concat([X_resampled, y_resampled], axis=1)
    return data

def dimensions(dataset_id, dataset):
    num_instances, num_features = dataset.shape 
    if dataset_id == "breast_cancer":
        num_instances, num_features = dataset.shape
    elif dataset_id == "hyperthyroidism":
        num_instances, num_features = dataset.shape
    elif dataset_id == "cervical_cancer":
        num_instances, num_features = dataset.shape
    elif dataset_id == "liver_cancer":
        num_instances, num_features = dataset.shape

    return num_instances, num_features

def feature_values(dataset_id, dataset):
    fvalues = []
    for col in dataset.columns:
        if dataset[col].dtype == 'float64':
            min_val = float(dataset[col].min()) 
            mean_val = float(dataset[col].mean())  
            max_val = float(dataset[col].max())  
            fvalues.append((min_val, mean_val, max_val))
        else:
            unique_values = set(dataset[col])
            fvalues.append(unique_values)
    return fvalues

#Get unique outcome values for all the dataset
def outcome_values(dataset_id, dataset):
    values = set()
    if dataset_id == "breast_cancer":
        target_variable = 'Diagnosis'
        unique_values = set(dataset[target_variable].unique())
        values.update(unique_values)
    if dataset_id == "hyperthyroidism":
        target_variable = 'classes'  
        unique_values = set(dataset[target_variable].unique())
        values.update(unique_values)
    if dataset_id == "cervical_cancer":
        target_variable = 'Biopsy'
        unique_values = set(str(value) for value in dataset[target_variable].unique())
        values.update(unique_values)
    if dataset_id == "liver_cancer":
        target_variable = 'classes' 
        unique_values = set(str(int(value)) for value in dataset[target_variable].unique())
        values.update(unique_values)
    return values

#Get the list of outcome values
def outcomes(dataset_id, instances):
    if dataset_id == "breast_cancer":
        target_variable = 0  # 'Diagnosis'
        target_values = np.array(instances)[:, target_variable].tolist()
    elif dataset_id == "hyperthyroidism":
        target_variable = -1
        target_values = np.array(instances)[:, target_variable].tolist()
    elif dataset_id == "cervical_cancer":
        target_variable = -1  # 'Biopsy'
        target_values = [str(value) for value in np.array(instances)[:, target_variable].tolist()]
    elif dataset_id == "liver_cancer":
        target_variable = -1  # 'classes'
        target_values = [str(value) for value in np.array(instances)[:, target_variable].tolist()]
    return target_values

#Split the concatenated dataset into train and test
def data_split(dataset_id, dataset, percent_train):
    num_instances = dataset.shape[0]
    num_train = int(num_instances * percent_train)
    train_data = dataset[:num_train]
    test_data = dataset[num_train:]
    train_data = train_data.values.tolist()
    test_data = test_data.values.tolist()   
    return train_data, test_data

#Get the most frequent class for each dataset
from collections import Counter

def baseline(dataset_id, dataset):
    if dataset_id == "breast_cancer":
        target_column = 0
        target_values = dataset.iloc[:, target_column]
        counts = Counter(target_values)
        most_frequent_class = str(counts.most_common(1)[0][0])
        return most_frequent_class
    elif dataset_id == "hyperthyroidism":
        target_column = 'classes'  
        target_values = dataset[target_column]
        counts = Counter(target_values)
        most_frequent_class = counts.most_common(1)[0][0]
        return most_frequent_class
    elif dataset_id == "cervical_cancer":
        target_column = -1
        target_values = dataset.iloc[:, target_column]
        counts = Counter(target_values)
        most_frequent_class = str(counts.most_common(1)[0][0])
        return str(most_frequent_class)
    elif dataset_id == "liver_cancer":
        target_column = -1
        target_values = dataset.iloc[:, target_column]
        counts = Counter(target_values)
        most_frequent_class = str(counts.most_common(1)[0][0])
        return str(most_frequent_class)
     
#Apply OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def preprocess_data(data, dataset_id):
    categorical_columns = []
    if dataset_id == "breast_cancer":
        categorical_columns = ['Diagnosis']
    elif dataset_id == "hyperthyroidism":
        categorical_columns = ['sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                               'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
                               'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured',
                               'TBG measured', 'referall source', 'classes']
    elif dataset_id == "cervical_cancer":
        categorical_columns = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                               'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes',
                               'STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'Biopsy']
    elif dataset_id == "liver_cancer":
        categorical_columns = ['classes']
    data.dropna(inplace=True)
    if categorical_columns:
        encoder = OneHotEncoder(sparse=False) 
        encoded_data = encoder.fit_transform(data[categorical_columns])
        encoded_columns = encoder.get_feature_names_out(input_features=categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
        data = data.drop(categorical_columns, axis=1)
        data = pd.concat([data, encoded_df], axis=1)
    numeric_columns = [col for col in data.columns if col not in categorical_columns]
    for col in numeric_columns:
        imputer = SimpleImputer(strategy='mean')
        data[col] = imputer.fit_transform(data[[col]])
    return data

#Apply Decision Tree
from sklearn.preprocessing import StandardScaler
def decision_tree(dataset_id, train, test):
    predictions = []  
    if dataset_id == "breast_cancer":
        y_train = [item[0] for item in train]  
        X_train = [item[1:] for item in train]  
        y_test = [item[0] for item in test]  
        X_test = [item[1:] for item in test]  
    elif dataset_id == "hyperthyroidism":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "cervical_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "liver_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test]  
    imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf_tree = DecisionTreeClassifier(max_depth=5, random_state=1)
    clf_tree.fit(X_train, y_train)
    predictions = clf_tree.predict(X_test)
    return [str(prediction) for prediction in predictions] 

#Apply knn
def knn(dataset_id, train, test):
    predictions = []

    if dataset_id == "breast_cancer":
        y_train = [item[0] for item in train]  
        X_train = [item[1:] for item in train]  
        y_test = [item[0] for item in test] 
        X_test = [item[1:] for item in test]  
    elif dataset_id == "hyperthyroidism":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test] 
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "cervical_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "liver_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test] 
    
    imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf_knn = KNeighborsClassifier(n_neighbors=7)
    clf_knn.fit(X_train, y_train)
    predictions = clf_knn.predict(X_test)

    return [str(prediction) for prediction in predictions] 

#Apply GaussianNB and XGBClassifier 
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
def naive_bayes(dataset_id, train, test):
    predictions = []
    if dataset_id == "breast_cancer":
            y_train = [item[0] for item in train]  
            X_train = [item[1:] for item in train]  
            y_test = [item[0] for item in test] 
            X_test = [item[1:] for item in test]  
    elif dataset_id == "hyperthyroidism":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test] 
            X_test = [item[:-1] for item in test]  
    elif dataset_id == "cervical_cancer":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test]  
            X_test = [item[:-1] for item in test]  
    elif dataset_id == "liver_cancer":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test]  
            X_test = [item[:-1] for item in test] 
    imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 
    clf_nb = GaussianNB()
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
    ensemble = VotingClassifier(estimators=[('nb', clf_nb), ('xgb', xgb)], voting='soft')
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    
    return [str(prediction) for prediction in predictions]

#Apply SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
def svm(dataset_id, train, test):
    predictions = []
    if dataset_id == "breast_cancer":
        y_train = [item[0] for item in train]  
        X_train = [item[1:] for item in train]  
        y_test = [item[0] for item in test] 
        X_test = [item[1:] for item in test]  
    elif dataset_id == "hyperthyroidism":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test] 
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "cervical_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test]  
    elif dataset_id == "liver_cancer":
        y_train = [item[-1] for item in train]  
        X_train = [item[:-1] for item in train]  
        y_test = [item[-1] for item in test]  
        X_test = [item[:-1] for item in test] 
    imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf_svc = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.01, kernel='linear', max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001, verbose=False)
    clf_svc.fit(X_train, y_train)
    predictions = clf_svc.predict(X_test)
    return [str(prediction) for prediction in predictions] 

#Get Accuracy, Precision, Recall, F1
import numpy as np
np.arange(5) == np.arange(5).astype(str)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def evaluate(dataset_id, gold, predictions):
    results = {}
    
    try:
        predictions = [str(i) for i in predictions]
    except ValueError:
        pass
    accuracy = accuracy_score(gold, predictions)
    results['accuracy'] = float(accuracy)
    precision = precision_score(gold, predictions, average='weighted', labels=list(set(gold)))
    recall = recall_score(gold, predictions, average='weighted', labels=list(set(gold)))
    f1 = f1_score(gold, predictions, average='weighted', labels=list(set(gold)))
    
    for i, output in enumerate(list(set(gold))):
        results[output] = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
        
    return results

#Get Learning Curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
def learning_curve(dataset_id, train_sets, test, class_func):
    accuracies = []
    for train in train_sets:
        if dataset_id == "breast_cancer":
            y_train = [item[0] for item in train]  
            X_train = [item[1:] for item in train]  
            y_test = [item[0] for item in test] 
            X_test = [item[1:] for item in test]  
        elif dataset_id == "hyperthyroidism":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test] 
            X_test = [item[:-1] for item in test]  
        elif dataset_id == "cervical_cancer":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test]  
            X_test = [item[:-1] for item in test]  
        elif dataset_id == "liver_cancer":
            y_train = [item[-1] for item in train]  
            X_train = [item[:-1] for item in train]  
            y_test = [item[-1] for item in test]  
            X_test = [item[:-1] for item in test] 
        lr = class_func(dataset_id, train, test)
        accuracies.append(float(accuracy_score(y_test,lr)))
    return accuracies

#Plot Learning Curve
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def visualize(dataset_id, train_sets, test, class_func):
        accuracies = [] 
        train_sizes = []  
        for train in train_sets:
            if dataset_id == "breast_cancer":
                y_train = [item[0] for item in train]  
                X_train = [item[1:] for item in train]  
                y_test = [item[0] for item in test] 
                X_test = [item[1:] for item in test]  
            elif dataset_id == "hyperthyroidism":
                y_train = [item[-1] for item in train]  
                X_train = [item[:-1] for item in train]  
                y_test = [item[-1] for item in test] 
                X_test = [item[:-1] for item in test]  
            elif dataset_id == "cervical_cancer":
                y_train = [item[-1] for item in train]  
                X_train = [item[:-1] for item in train]  
                y_test = [item[-1] for item in test]  
                X_test = [item[:-1] for item in test]  
            elif dataset_id == "liver_cancer":
                y_train = [item[-1] for item in train]  
                X_train = [item[:-1] for item in train]  
                y_test = [item[-1] for item in test]  
                X_test = [item[:-1] for item in test]
            lr = class_func(dataset_id, train, test)
            accuracy = float(accuracy_score(y_test, lr))
            accuracies.append(accuracy)
            train_sizes.append(len(train))  
            plt.figure()
            plt.title(f'Learning Curve for {dataset_id} using {class_func.__name__}')
            plt.xlabel("Training Instances")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.plot(train_sizes, accuracies, 'o-', color="r", label="Accuracy")
            plt.legend(loc="best")
                
            desktop_path = os.path.expanduser("~") 
            file_name = f"learning_curve_{dataset_id}_{class_func.__name__}.png"
            file_path = os.path.join(desktop_path, "Desktop", file_name)
            plt.savefig(file_path)
            plt.show()

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  datasets = ['breast_cancer',\
              'hyperthyroidism',\
              'cervical_cancer',\
              'liver_cancer']
  dims =    {'breast_cancer':(569, 30),
             'hyperthyroidism':(3772, 29),
             'cervical_cancer':(858, 28),
             'liver_cancer':(583,10)}
  targets = {'breast_cancer':set(['B', 'M']),
             'hyperthyroidism':set(['goitre', 'secondary toxic', 'negative', 'T3 toxic', 'hyperthyroid']),
             'cervical_cancer':set(['0', '1']),
             'liver_cancer':set(['1', '2'])}

  for dataset_id in datasets:
    print('::  DATASET: %s ::' % dataset_id)
    print('::: Problem 0-A :::')
    data = read_data(dataset_id)
    util_5353.assert_not_none(data, '0-A')
    
    print('::: Problem 0-D :::')
    d_ret = outcome_values(dataset_id, data)
    util_5353.assert_set(d_ret, '0-D', valid_values=targets[dataset_id])
    print(d_ret)

    print('::: Problem 1 :::')
    one_ret = data_split(dataset_id, data, 0.6)
    util_5353.assert_tuple(one_ret, 2, '1')
    util_5353.assert_list(one_ret[0], None, '1')
    util_5353.assert_list(one_ret[1], None, '1')
    if dataset_id == 'breast_cancer':
      util_5353.assert_list(one_ret[0], 341, '1')
    if dataset_id == 'cervical_cancer':
      util_5353.assert_list(one_ret[0], 514, '1')
    train = one_ret[0]
    test  = one_ret[1]
    
    print('::: Problem 0-E :::')
    train_out = outcomes(dataset_id, train)
    test_out  = outcomes(dataset_id, test)
    util_5353.assert_list(train_out, len(train), '0-E', valid_values=targets[dataset_id])
    util_5353.assert_list(test_out,  len(test),  '0-E', valid_values=targets[dataset_id])
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('M', train_out[0], '0-E')
      util_5353.assert_str_eq('B', test_out[-1], '0-E')
      
    print('::: Problem 2 :::')
    two_ret = baseline(dataset_id, data)
    util_5353.assert_str(two_ret, '2')
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('B', two_ret, '2')
    print(two_ret)
    preprocessed_data = preprocess_data(data, dataset_id)
    print('::: Problem 3 :::')
    three_ret = decision_tree(dataset_id, train, test)
    util_5353.assert_list(three_ret, len(test), '3')
    
    print('::: Problem 4 :::')
    four_ret = knn(dataset_id, train, test)
    util_5353.assert_list(four_ret, len(test), '4')
    #print(four_ret)
    print('::: Problem 5 :::')
    five_ret = naive_bayes(dataset_id, train, test)
    util_5353.assert_list(five_ret, len(test), '5')
    #print(five_ret)
    print('::: Problem 6 :::')
    six_ret = svm(dataset_id, train, test)
    util_5353.assert_list(six_ret, len(test), '6')
    #print(six_ret)
        
    print('::: Problem 7 :::')
    seven_ret_dt = evaluate(dataset_id, test_out, three_ret)
    seven_ret_kn = evaluate(dataset_id, test_out, four_ret)
    seven_ret_nb = evaluate(dataset_id, test_out, five_ret)
    seven_ret_sv = evaluate(dataset_id, test_out, six_ret)
    for seven_ret in [seven_ret_dt, seven_ret_kn, seven_ret_nb, seven_ret_sv]:
      util_5353.assert_dict(seven_ret, '7')
      util_5353.assert_dict_key(seven_ret, 'accuracy', '7')
      util_5353.assert_float(seven_ret['accuracy'], '7')
      util_5353.assert_float_range((0.0, 1.0), seven_ret['accuracy'], '7')
      for target in targets[dataset_id]:
        util_5353.assert_dict_key(seven_ret, target, '7')
        util_5353.assert_dict(seven_ret[target], '7')
        util_5353.assert_dict_key(seven_ret[target], 'precision', '7')
        util_5353.assert_dict_key(seven_ret[target], 'recall', '7')
        util_5353.assert_dict_key(seven_ret[target], 'f1', '7')
        util_5353.assert_float(seven_ret[target]['precision'], '7')
        util_5353.assert_float(seven_ret[target]['recall'], '7')
        util_5353.assert_float(seven_ret[target]['f1'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['precision'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['recall'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['f1'], '7')
    print(seven_ret_dt)
    print(seven_ret_kn)
    print(seven_ret_nb)
    print(seven_ret_sv)
    
    print('::: Problem 8 :::')
    train_sets = []
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        train_sets.append(train[:int(percent*len(train))])
    eight_ret_dt = learning_curve(dataset_id, train_sets, test, decision_tree)
    eight_ret_kn = learning_curve(dataset_id, train_sets, test, knn)
    eight_ret_nb = learning_curve(dataset_id, train_sets, test, naive_bayes)
    eight_ret_sv = learning_curve(dataset_id, train_sets, test, svm)
    for eight_ret in [eight_ret_dt, eight_ret_kn, eight_ret_nb, eight_ret_sv]:
        util_5353.assert_list(eight_ret, len(train_sets), '8')
        for i in range(len(eight_ret)):
            util_5353.assert_float(eight_ret[i], '8')
            util_5353.assert_float_range((0.0, 1.0), eight_ret[i], '8')
    #visualize(dataset_id, train_sets, test, decision_tree)
    print(eight_ret_dt)
    print(eight_ret_kn)
    print(eight_ret_nb)
    print(eight_ret_sv)
    #dataset_ids = ["breast_cancer", "hyperthyroidism", "cervical_cancer", "liver_cancer"]
    visualize(dataset_id, train_sets, test, decision_tree)
    visualize(dataset_id, train_sets, test, knn)
    visualize(dataset_id, train_sets, test, naive_bayes)
    visualize(dataset_id, train_sets, test, svm)
print('~~~ All Tests Pass ~~~')



