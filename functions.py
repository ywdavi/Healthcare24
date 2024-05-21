import streamlit as st
import streamlit_authenticator as stauth

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydicom import dcmread
import glob
import radiomics
from radiomics import featureextractor, getTestCase
import SimpleITK as sitk
import os
import cv2


from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score

from scipy.stats import mode
import scipy.stats as stats


from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score

from sklearn.metrics import RocCurveDisplay, auc

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# This function gets as inputs an image and a mask, of type 'SimpleITK.SimpleITK.Image', and
# the list of radiomic features we are interested in, it returns a dataset with the features
# of the whole images.

def featurexImg(image, mask, ftype):
    df = pd.DataFrame()
    for feat in ftype:
        features_list = []
        # Define the feature extractor we're going to employ
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName(feat)

        # img = sitk.ReadImage(image_path, sitk.sitkFloat32)
        # msk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        features = extractor.execute(image, mask, label=1)
        og_features = {key: value.item() for key, value in features.items() if key.startswith('original_')}
        features_list.append(og_features)

        # convert the list in a pd.DataFrame and returns it
        curr_df = pd.DataFrame(features_list)
        df = pd.concat([df, curr_df], axis=1)
    return df


def load_image(image, size):
  '''
  reads, rezises, converts to grayscale and normalizes the input path
  '''
  # image = cv2.imread(path)
  image = cv2.resize(image, (size,size))
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = image/255.
  return image


# Definitions of the metrics


def TP(prediction:list, test:list)->int:                                 # considering 1 as positive class
     count = 0
     for x in range(len(prediction)):
         if prediction[x]==test[x] and test[x]==1:
             count+=1
     return count


def TN(prediction:list, test:list)->int:                                 # considering 0 as negative class
    count = 0
    for x in range(len(prediction)):
        if prediction[x]==test[x] and test[x]==0:
            count+=1
    return count


def accuracy(prediction:list, test:list)->int:                           # (TP+TN) / n
    return round((TP(prediction, test)+TN(prediction,test))/len(test),3)


def sensitivity(prediction:list, test:list)->int:                        # TP / TP + FN
    return round(TP(prediction, test)/test.count(1),3)


def specificity(prediction:list, test:list)->int:                        # TN / TN + FP
    return round(TN(prediction, test)/test.count(0),3)


def ci(mean, std, n):                                                    # 95% confidence interval
    z = stats.norm.ppf(0.975)
    margin_of_error = z * (std / (n ** 0.5))
    lower_bound = round((mean - margin_of_error), 3)
    upper_bound = round((mean + margin_of_error), 3)

    return lower_bound, upper_bound

"""
df = pd.read_csv("features.csv")

# Train-Test split
rs = 0
y = df["tumor"]
X = df.drop(['tumor'], axis=1)

X_train, X_score, y_train, y_score = train_test_split(X, y, test_size=0.3, stratify=y, random_state=rs)

# 3. Z-score normalization on respective subsets
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_score = pd.DataFrame(scaler.fit_transform(X_score), columns=X_score.columns)

K = 5

classifier = SVC(random_state=rs)
cv = KFold(n_splits = K, shuffle = True, random_state = rs)

param_grid = {
    'C': [1, 5, 10, 20, 30],
    'kernel': ['linear', 'rbf', 'poly']
}

# Perform nested cross-validation with grid search for hyperparameter tuning
grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
svm_classifier = grid_search.best_estimator_

# Evaluate the model
nested_scores = cross_val_score(svm_classifier, X_train, y_train, cv=cv)
#print(nested_scores)

# Print the results
print("The best cross-validated hyperparameters are the following:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print("\nNested CV accuracy with tuned hyperparameters: %.2f%% (+/- %.2f%%)" % (nested_scores.mean() * 100, nested_scores.std() * 100))


predicted = svm_classifier.predict(X_score)

print(f"Test accuracy: {accuracy(predicted.tolist(), y_score.tolist())}")
print(f"Test sensitivity {sensitivity(predicted.tolist(), y_score.tolist())}")
print(f"Test specificity {specificity(predicted.tolist(), y_score.tolist())}")

# ROC Curve with AUC
plt.figure(figsize=(10, 10), dpi=100)
svc_disp = RocCurveDisplay.from_estimator(svm_classifier, X_score, y_score, alpha=0.8, lw=2, color="b")  # ROC
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Chance level (AUC=0.50)") # alternative chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC curve \n(Positive label {4})')

plt.legend(loc="lower right")

# plt.show()

# Confusion Matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots

color_map = "PuBu"

# Plot confusion matrix with absolute values
axs[0].set_title('Confusion Matrix (Absolutes)')
CM_absolute = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_score.tolist(), predicted, labels=[1,0]),
    display_labels=["malignant","benign"])
CM_absolute.plot(cmap=color_map, ax=axs[0])

# Plot confusion matrix with percentages
axs[1].set_title('Confusion Matrix (Percentage)')
conf_matrix = confusion_matrix(y_score, predicted, labels=[1,0])
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
CM_percentage = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percentage, display_labels=["malignant","benign"])
CM_percentage.plot(cmap=color_map, ax=axs[1])

plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

"""



## -------------------- UI -------------------------- ##

# Login page

# Main page:
    # - column1: load file
    # - column2: display image

    # segment image (manually or automatically) --> create mask (display it!)

    # extract pyradiomics features --> feed them to the model for prediction

    # display prediction


    # reset button to manually segment again?

