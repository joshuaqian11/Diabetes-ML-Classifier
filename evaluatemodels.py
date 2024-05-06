import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def main():
    x = pd.read_csv('diabetes_x.csv')
    y = pd.read_csv('diabetes_y.csv').values.ravel()
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes) if len(classes) > 2 else y

    # split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_test_bin = label_binarize(y_test, classes=classes) if len(classes) > 2 else y_test

    # Define models and parameter grids
    models = {
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }),
        'Decision Tree': (DecisionTreeClassifier(), {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }),
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [50, 100, 200],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
        }),
        'Naive Bayes': (GaussianNB(), {
            'var_smoothing': [1e-09, 1e-08, 1e-07]
        })
    }

    best_params = {}

    # Perform grid search and collect results
    for name, (model, params) in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params[name] = grid_search.best_params_
        print(f"Best Parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation score for {name}: {grid_search.best_score_:.4f}\n")

    best_models = {
        'KNN': KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='uniform'),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=2),
        'Random Forest': RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=50),
        'Naive Bayes': GaussianNB(var_smoothing=1e-07)
    }

    results = {}
    roc_data = {}

    # train each model with the best parameters and evaluate
    # iterate through all models
    for name, model in best_models.items():
        # Train the models
        model.fit(x_train, y_train)
        # Predict values
        predictions = model.predict(x_test)
        # Assuming Y = 1 is positive class
        probabilities = model.predict_proba(x_test)[:, 1]

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            roc_auc = auc(fpr, tpr)
        else:
            # for multi-class, compute ROC AUC for each class and take average
            fpr, tpr, roc_auc = {}, {}, {}
            for i, class_ in enumerate(classes):
                fpr[class_], tpr[class_], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
                roc_auc[class_] = auc(fpr[class_], tpr[class_])
            roc_auc = np.mean(list(roc_auc.values()))

        # build results
        results[name] = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average='weighted'),
            'Recall': recall_score(y_test, predictions, average='weighted'),
            'F1 Score': f1_score(y_test, predictions, average='weighted'),
            'AUC': roc_auc
        }

        roc_data[name] = (fpr, tpr)

    for model, metrics in results.items():
        print(f"{model} Model Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print()

    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, (model, (fpr, tpr)) in enumerate(roc_data.items()):
        if len(classes) == 2:
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{model} ROC curve (area = {results[model]["AUC"]:.2f})')
        else:
            for class_ in classes:
                plt.plot(fpr[class_], tpr[class_], color=colors[(i * len(classes) + class_) % len(colors)],
                         lw=2, label=f'{model} ROC curve (area = {results[model]["AUC"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
