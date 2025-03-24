# %%
import os
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,recall_score,f1_score,precision_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import skops.io as sio

# %%
'''
## Image preprocessing
'''

# %%
# Define Image Transformations RGB
def load_images_to_array(directory):
    images,imagesRGB,imagesHSV, labels,labelsRGB =[], [], [],[],[]
    for i,dir in enumerate(directory):
        for file in os.listdir(dir):
            img_path = os.path.join(dir, file)
            img = Image.open(img_path)  
            img = img.resize((128, 128))
            images.append(np.array(img.convert('L')).flatten())
            labels.append(i)
            if img.mode == 'RGB':
                imagesRGB.append(np.array(img).flatten())
                imagesHSV.append(np.array(img.convert('HSV')).flatten())
                labelsRGB.append(i)
                
    return np.array(images),np.array(imagesRGB),np.array(imagesHSV), np.array(labels),np.array(labelsRGB)

# Define SIFT Feature Extraction
def extract_bovw_features(directory,num_clusters=50):
    sift = cv2.SIFT_create()
    images, labels = [], []
    for i,dir in enumerate(directory):
        for file in os.listdir(dir):
            img_path = os.path.join(dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                images.append(descriptors)
                labels.append(i)
        # Stack all descriptors
    descriptors_stack = np.vstack(images)

    # Step 2: Apply KMeans clustering to form visual words
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(descriptors_stack)
    sio.dump(kmeans, "./Model/kmeanss.skops")
    # Step 3: Create histograms for each image
    bovw_features = []

    for descriptors in images:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=num_clusters, range=(0, num_clusters))
        bovw_features.append(hist)
        
        
    return np.array(bovw_features), np.array(labels)
    
# Load Dataset
data_dir_train = ["Training/museum-indoor","Training/museum-outdoor" ] 
data_dir_test = ["Museum_Validation/museum-indoor","Museum_Validation/museum-outdoor" ] 

X_train,X_train_rgb,X_train_hsv,y_train,y_train_rgb = load_images_to_array(data_dir_train)
X_test,X_test_rgb,X_test_hsv,y_test,y_test_rgb = load_images_to_array(data_dir_test)


X_train_sift, y_train_sift = extract_bovw_features(data_dir_train)
X_test_sift, y_test_sift = extract_bovw_features(data_dir_test)

# %%
'''
# Models
'''

# %%
'''
## Decision Tree Classifier
'''

# %%
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Standardize features
    ('pca', PCA()),   # PCA without specifying components initially
    ('dt', DecisionTreeClassifier(random_state=42))
])

# Define hyperparameter grid
param_grid = {
   'pca__n_components': [150],
    'dt__max_depth': [6],
    'dt__min_samples_split': [4],
    'dt__min_samples_leaf': [1],
}


# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
dt = grid_search.best_estimator_

grid_search.fit(X_train_rgb, y_train_rgb)
dt_rgb = grid_search.best_estimator_

grid_search.fit(X_train_hsv, y_train_rgb)
dt_hsv = grid_search.best_estimator_


#Define hyperparameter grid SIFT
param_grid = {
    'dt__max_depth': [6],
    'dt__min_samples_split': [4],
    'dt__min_samples_leaf': [1],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train_sift, y_train_sift)
dt_sift = grid_search.best_estimator_

# %%
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("Decision Tree Recall:", recall_score(y_test, y_pred_dt, average='weighted'))
print("Decision Tree F1 Score:", f1_score(y_test, y_pred_dt, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt),display_labels=dt.classes_).plot()
print(classification_report(y_test, y_pred_dt))

# %%
y_pred_dt_rgb = dt_rgb.predict(X_test_rgb)
print("Decision Tree Accuracy:", accuracy_score(y_test_rgb, y_pred_dt_rgb))
print("Decision Tree Precision:", precision_score(y_test_rgb, y_pred_dt_rgb, average='weighted'))
print("Decision Tree Recall:", recall_score(y_test_rgb, y_pred_dt_rgb, average='weighted'))
print("Decision Tree F1 Score:", f1_score(y_test_rgb, y_pred_dt_rgb, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_rgb, y_pred_dt_rgb),display_labels=dt_rgb.classes_).plot()
print(classification_report(y_test_rgb, y_pred_dt_rgb))

# %%
y_pred_dt_hsv = dt_hsv.predict(X_test_hsv)
print("Decision Tree Accuracy:", accuracy_score(y_test_rgb, y_pred_dt_hsv))
print("Decision Tree Precision:", precision_score(y_test_rgb, y_pred_dt_hsv, average='weighted'))
print("Decision Tree Recall:", recall_score(y_test_rgb, y_pred_dt_hsv, average='weighted'))
print("Decision Tree F1 Score:", f1_score(y_test_rgb, y_pred_dt_hsv, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_rgb, y_pred_dt_hsv),display_labels=dt_hsv.classes_).plot()
print(classification_report(y_test_rgb, y_pred_dt_hsv))

# %%
y_pred_dt = dt_sift.predict(X_test_sift)
print("Decision Tree Accuracy:", accuracy_score(y_test_sift, y_pred_dt))
print("Decision Tree Precision:", precision_score(y_test_sift, y_pred_dt, average='weighted'))
print("Decision Tree Recall:", recall_score(y_test_sift, y_pred_dt, average='weighted'))
print("Decision Tree F1 Score:", f1_score(y_test_sift, y_pred_dt, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_sift, y_pred_dt),display_labels=dt_sift.classes_).plot()
print(classification_report(y_test_sift, y_pred_dt))

sio.dump(dt_rgb, "./Model/dt_rgb.skops")
sio.dump(dt_hsv, "./Model/dt_hsv.skops")
sio.dump(dt_sift, "./Model/dt_sift.skops")

# %%
'''
## Random Forest Classifier
'''

# %%
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Standardize features
    ('pca', PCA()),   # PCA without specifying components initially
    ('rf', RandomForestClassifier(random_state=42))  # Random Forest
])

# Define hyperparameter grid
param_grid = {
    'pca__n_components': [150, 160],  # Tune PCA components
    'rf__n_estimators': [50, 100],  # Number of trees
    'rf__max_depth': [None, 10]  # Maximum depth of trees
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
rf = grid_search.best_estimator_

grid_search.fit(X_train_rgb, y_train_rgb)
rf_rgb = grid_search.best_estimator_

grid_search.fit(X_train_hsv, y_train_rgb)
rf_hsv = grid_search.best_estimator_

#Define hyperparameter grid SIFT
param_grid = {
  'rf__n_estimators': [50, 100],  # Number of trees
    'rf__max_depth': [None, 10]  # Maximum depth of trees
}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train_sift, y_train_sift)
rf_sift = grid_search.best_estimator_

# %%
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Random Forest Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),display_labels=rf.classes_).plot()
print(classification_report(y_test, y_pred_rf))

# %%
y_pred_rf_rgb = rf_rgb.predict(X_test_rgb)
print("Random Forest Accuracy:", accuracy_score(y_test_rgb, y_pred_rf_rgb))
print("Random Forest Precision:", precision_score(y_test_rgb, y_pred_rf_rgb, average='weighted'))
print("Random Forest Recall:", recall_score(y_test_rgb, y_pred_rf_rgb, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test_rgb, y_pred_rf_rgb, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_rgb, y_pred_rf_rgb),display_labels=rf_rgb.classes_).plot()
print(classification_report(y_test_rgb, y_pred_rf_rgb))

# %%
y_pred_rf_hsv = rf_hsv.predict(X_test_hsv)
print("Random Forest Accuracy:", accuracy_score(y_test_rgb, y_pred_rf_hsv))
print("Random Forest Precision:", precision_score(y_test_rgb, y_pred_rf_hsv, average='weighted'))
print("Random Forest Recall:", recall_score(y_test_rgb, y_pred_rf_hsv, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test_rgb, y_pred_rf_hsv, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_rgb, y_pred_rf_hsv),display_labels=rf_hsv.classes_).plot()
print(classification_report(y_test_rgb, y_pred_rf_hsv))

# %%
y_pred_rf = rf_sift.predict(X_test_sift)
print("Random Forest Accuracy:", accuracy_score(y_test_sift, y_pred_rf))
print("Random Forest Precision:", precision_score(y_test_sift, y_pred_rf, average='weighted'))
print("Random Forest Recall:", recall_score(y_test_sift, y_pred_rf, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test_sift, y_pred_rf, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix(y_test_sift, y_pred_rf),display_labels=rf_sift.classes_).plot()
print(classification_report(y_test_sift, y_pred_rf))
sio.dump(rf_rgb, "./Model/rf_rgb.skops")
sio.dump(rf_hsv, "./Model/rf_hsv.skops")
sio.dump(rf_sift, "./Model/rf_sift.skops")
# %%
'''
## Gradient Boosting Classifier
'''

# %%
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Standardize features
    ('pca', PCA()),   # PCA without specifying components initially
    ('gb', XGBClassifier(random_state=42))  # XGBoost
])

# Define hyperparameter grid
param_grid = {
    'pca__n_components': [150, 160],  # Tune PCA components
    'gb__n_estimators': [50, 100],  # Number of boosting stages
    'gb__learning_rate': [0.01, 0.1]  # Learning rate
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
gb = grid_search.best_estimator_

grid_search.fit(X_train_rgb, y_train_rgb)
gb_rgb = grid_search.best_estimator_

grid_search.fit(X_train_hsv, y_train_rgb)
gb_hsv = grid_search.best_estimator_

#Define hyperparameter grid SIFT
param_grid = {
    'gb__n_estimators': [50, 100],  # Number of boosting stages
    'gb__learning_rate': [0.01, 0.1]  # Learning rate
}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')

grid_search.fit(X_train_sift, y_train_sift)
gb_sift = grid_search.best_estimator_

# %%
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Precision:", precision_score(y_test, y_pred_gb, average='weighted'))
print("Gradient Boosting Recall:", recall_score(y_test, y_pred_gb, average='weighted'))
print("Gradient Boosting F1 Score:", f1_score(y_test, y_pred_gb, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_gb),display_labels=gb.classes_).plot()
print(classification_report(y_test, y_pred_gb))

# %%
y_pred_gb_rgb = gb_rgb.predict(X_test_rgb)
print("Gradient Boosting Accuracy:", accuracy_score(y_test_rgb, y_pred_gb_rgb))
print("Gradient Boosting Precision:", precision_score(y_test_rgb, y_pred_gb_rgb, average='weighted'))
print("Gradient Boosting Recall:", recall_score(y_test_rgb, y_pred_gb_rgb, average='weighted'))
print("Gradient Boosting F1 Score:", f1_score(y_test_rgb, y_pred_gb_rgb, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_rgb, y_pred_gb_rgb),display_labels=gb_rgb.classes_).plot()
print(classification_report(y_test_rgb, y_pred_gb_rgb))

# %%
y_pred_gb_hsv = gb_hsv.predict(X_test_hsv)
print("Gradient Boosting Accuracy:", accuracy_score(y_test_rgb, y_pred_gb_hsv))
print("Gradient Boosting Precision:", precision_score(y_test_rgb, y_pred_gb_hsv, average='weighted'))
print("Gradient Boosting Recall:", recall_score(y_test_rgb, y_pred_gb_hsv, average='weighted'))
print("Gradient Boosting F1 Score:", f1_score(y_test_rgb, y_pred_gb_hsv, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_rgb, y_pred_gb_hsv),display_labels=gb_hsv.classes_).plot()
print(classification_report(y_test_rgb, y_pred_gb_hsv))

# %%
y_pred_gb = gb_sift.predict(X_test_sift)
print("Gradient Boosting Accuracy:", accuracy_score(y_test_sift, y_pred_gb))
print("Gradient Boosting Precision:", precision_score(y_test_sift, y_pred_gb, average='weighted'))
print("Gradient Boosting Recall:", recall_score(y_test_sift, y_pred_gb, average='weighted'))
print("Gradient Boosting F1 Score:", f1_score(y_test_sift, y_pred_gb, average='weighted'))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_sift, y_pred_gb),display_labels=gb_sift.classes_).plot()
print(classification_report(y_test_sift, y_pred_gb))

sio.dump(gb_rgb, "./Model/gb_rgb.skops")
sio.dump(gb_hsv, "./Model/gb_hsv.skops")
sio.dump(gb_sift, "./Model/gb_sift.skops")
