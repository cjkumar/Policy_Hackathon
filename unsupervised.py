import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'bold_dataset.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Feature engineering: Calculate delta between SaO2 and SpO2
data['Delta_SaO2_SpO2'] = data['SaO2'] - data['SpO2']

# Create the target variable: Hypoxic (<94) or Non-Hypoxic (≥94)
data['SaO2_Category'] = np.where(data['SaO2'] < 94, 0, 1)  # 0 = Hypoxic, 1 = Non-Hypoxic

# Include race/ethnicity as a categorical feature and preprocess it
categorical_features = ['race_ethnicity']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_race_ethnicity = encoder.fit_transform(data[categorical_features])

# Convert encoded categories to a DataFrame with proper column names
encoded_race_ethnicity_df = pd.DataFrame(
    encoded_race_ethnicity, columns=encoder.get_feature_names_out(categorical_features)
)

# Select features for prediction
numeric_features = [
    'admission_age', 'sex_female', 'BMI_admission'
]
target = 'SaO2_Category'  # New target variable

# Filter dataset for selected features and target
data_subset = pd.concat([data[numeric_features], encoded_race_ethnicity_df, data[target]], axis=1)

# Drop rows with missing target values
data_subset = data_subset.dropna(subset=[target])

# Split data into features (X) and target (y)
X = data_subset.drop(columns=[target])
y = data_subset[target]

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 1. Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
clf.fit(X_train, y_train)

# Random Forest Predictions
y_pred_rf = clf.predict(X_test)
y_pred_rf_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Feature Importance Analysis
feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# 2. Train a Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Logistic Regression Predictions
y_pred_lr = log_reg.predict(X_test)
y_pred_lr_proba = log_reg.predict_proba(X_test)[:, 1]

# Logistic Regression Evaluation
accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr_proba)
classification_rep_lr = classification_report(y_test, y_pred_lr)

# Logistic Regression Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# 3. Perform PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained Variance by PCA Components
explained_variance = pca.explained_variance_ratio_

# Output Results
print("\n=== Random Forest ===")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"ROC-AUC Score: {roc_auc_rf:.2f}")
print("\nClassification Report:")
print(classification_rep_rf)
print("\nTop Features by Importance:")
print(importance_df.head(10))

print("\n=== Logistic Regression ===")
print(f"Accuracy: {accuracy_lr:.2f}")
print(f"ROC-AUC Score: {roc_auc_lr:.2f}")
print("\nClassification Report:")
print(classification_rep_lr)
print("\nLogistic Regression Coefficients:")
print(coefficients)

print("\n=== PCA ===")
print(f"Explained Variance (PC1 & PC2): {explained_variance}")

# Visualize PCA
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='SaO2 Category (0 = Hypoxic, 1 = Non-Hypoxic)')
plt.show()


from sklearn.cluster import KMeans

# K-Means Clustering for Hypoxic and Non-Hypoxic Groups
def perform_kmeans(X, n_clusters=3):
    """
    Perform K-Means clustering and return cluster labels and centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

# Split the data into hypoxic and non-hypoxic groups
hypoxic_data = data_subset[data_subset['SaO2_Category'] == 0]
non_hypoxic_data = data_subset[data_subset['SaO2_Category'] == 1]

# Perform clustering for hypoxic patients
X_hypoxic = hypoxic_data[['admission_age', 'sex_female', 'BMI_admission']].dropna()
labels_hypoxic, centroids_hypoxic = perform_kmeans(X_hypoxic, n_clusters=3)

# Map cluster labels back to the original hypoxic_data index
hypoxic_data = hypoxic_data.copy()
hypoxic_data['Cluster'] = np.nan  # Initialize with NaN
hypoxic_data.loc[X_hypoxic.index, 'Cluster'] = labels_hypoxic  # Assign cluster labels

# Perform clustering for non-hypoxic patients
X_non_hypoxic = non_hypoxic_data[['admission_age', 'sex_female', 'BMI_admission']].dropna()
labels_non_hypoxic, centroids_non_hypoxic = perform_kmeans(X_non_hypoxic, n_clusters=3)

# Map cluster labels back to the original non_hypoxic_data index
non_hypoxic_data = non_hypoxic_data.copy()
non_hypoxic_data['Cluster'] = np.nan  # Initialize with NaN
non_hypoxic_data.loc[X_non_hypoxic.index, 'Cluster'] = labels_non_hypoxic  # Assign cluster labels


# Add cluster labels to the original dataset
hypoxic_data = hypoxic_data.copy()
non_hypoxic_data = non_hypoxic_data.copy()
hypoxic_data['Cluster'] = labels_hypoxic
non_hypoxic_data['Cluster'] = labels_non_hypoxic

# Combine both groups back into one dataset
data_with_clusters = pd.concat([hypoxic_data, non_hypoxic_data])

# Output the bounds (centroids) for each cluster
centroids_hypoxic_df = pd.DataFrame(centroids_hypoxic, columns=['admission_age', 'sex_female', 'BMI_admission'])
centroids_non_hypoxic_df = pd.DataFrame(centroids_non_hypoxic, columns=['admission_age', 'sex_female', 'BMI_admission'])

print("\n=== Hypoxic Patient Archetypes (Cluster Centroids) ===")
print(centroids_hypoxic_df)

print("\n=== Non-Hypoxic Patient Archetypes (Cluster Centroids) ===")
print(centroids_non_hypoxic_df)

# Visualize Clusters for Hypoxic Patients
plt.figure(figsize=(8, 6))
plt.scatter(X_hypoxic['admission_age'], X_hypoxic['BMI_admission'], c=labels_hypoxic, cmap='viridis', alpha=0.7)
plt.scatter(centroids_hypoxic[:, 0], centroids_hypoxic[:, 2], color='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering for Hypoxic Patients')
plt.xlabel('Admission Age')
plt.ylabel('BMI')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualize Clusters for Non-Hypoxic Patients
plt.figure(figsize=(8, 6))
plt.scatter(X_non_hypoxic['admission_age'], X_non_hypoxic['BMI_admission'], c=labels_non_hypoxic, cmap='viridis', alpha=0.7)
plt.scatter(centroids_non_hypoxic[:, 0], centroids_non_hypoxic[:, 2], color='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering for Non-Hypoxic Patients')
plt.xlabel('Admission Age')
plt.ylabel('BMI')
plt.legend()
plt.grid(alpha=0.3)
plt.show()