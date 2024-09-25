import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Load the raw dataset including 'Suburb'
df_raw = pd.read_csv('DataSets/combined_housing_data.csv', usecols=['Bedrooms', 'Type', 'Price', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb'])

# Function to split the data into features (X) and target (y)
def split_data(df, target='Price', classification=False):
    X = df[['Bedrooms', 'Type', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb']]
    y = df[target]
    
    # For classification, categorize the house prices into bins
    if classification:
        y = pd.qcut(y, q=3, labels=['low', 'medium', 'high'])
        
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline with scaling for models that benefit from it
def get_preprocessor(apply_scaling):
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type', 'Suburb'])  # Encode categorical columns
    ]
    
    if apply_scaling:
        transformers.append(('num', StandardScaler(), ['Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built']))
    
    return ColumnTransformer(transformers=transformers)

# Regression: Train and evaluate a linear regression model
def train_and_evaluate_linear_regression(X_train, X_test, y_train, y_test, apply_scaling=False):
    # Preprocess and fit the model
    preprocessor = get_preprocessor(apply_scaling)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('linear_regression', LinearRegression())])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Print the first 10 results and accuracy
    results = pd.DataFrame({'Actual Price': y_test[:10], 'Predicted Price': y_pred[:10]})
    print("\n--- Linear Regression Results (first 10 predictions) ---")
    print(results.to_string(index=False))
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.4f}")

    return y_test, y_pred

# Clustering: Apply K-Means and reduce dimensionality using PCA
def kmeans_clustering(X_train, X_test, X_test_raw, apply_scaling=False):
    # Preprocess data
    preprocessor = get_preprocessor(apply_scaling)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train_preprocessed)
    labels = kmeans.predict(X_test_preprocessed)
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_preprocessed)

    # Determine which clusters are "low", "medium", and "high" based on average price in clusters
    cluster_mean_prices = []
    for cluster_label in np.unique(labels):
        cluster_indices = labels == cluster_label
        mean_price = X_test_raw['Price'].iloc[cluster_indices].mean()
        cluster_mean_prices.append((cluster_label, mean_price))
    
    # Sort clusters by average price and map labels to descriptive categories
    sorted_clusters = sorted(cluster_mean_prices, key=lambda x: x[1])
    cluster_mapping = {sorted_clusters[0][0]: '1', sorted_clusters[1][0]: '2', sorted_clusters[2][0]: '3'}
    
    # Map the numeric cluster labels to descriptive labels
    mapped_labels = [cluster_mapping[label] for label in labels]
    
    # Print the first 10 labels
    print("\n--- K-Means Clustering Labels (first 10 labels) ---")
    print(mapped_labels[:10])

    return X_test_pca, mapped_labels

# Classification: Train and evaluate a K-Nearest Neighbors classifier
def train_and_evaluate_knn_classifier(X_train, X_test, y_train, y_test, apply_scaling=False):
    preprocessor = get_preprocessor(apply_scaling)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('knn_classifier', KNeighborsClassifier(n_neighbors=5))])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Print the first 10 classification results
    results = pd.DataFrame({'Actual Category': y_test[:10], 'Predicted Category': y_pred[:10]})
    print("\n--- K-Nearest Neighbors Classification Results (first 10 predictions) ---")
    print(results.to_string(index=False))
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return y_test, y_pred, accuracy

# Function to plot results with descriptive labels and legend, without the empty plot
def plot_results_combined(y_test, y_pred_lr, X_test_pca, kmeans_labels, y_test_knn, y_pred_knn):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot regression (Actual vs Predicted)
    axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.3)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_title("Linear Regression (Actual vs Predicted)")
    axes[0, 0].set_xlabel("Actual Prices")
    axes[0, 0].set_ylabel("Predicted Prices")
    
    #Colors for the clusters
    color_mapping = {
        '1': 'red',
        '2': 'green',
        '3': 'blue',
    } 

    # Map the labels to colors
    colors = [color_mapping[label] for label in kmeans_labels]

    # Plot clustering (K-Means with PCA)
    scatter = axes[0, 1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=colors, alpha=0.6)
    axes[0, 1].set_title("K-Means Clustering (PCA)")
    axes[0, 1].set_xlabel("PCA Component 1")
    axes[0, 1].set_ylabel("PCA Component 2")
    
    # Add a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                markerfacecolor=color_mapping[label], markersize=10) for label in color_mapping]
    axes[0, 1].legend(handles=handles, title="Cluster", loc="best")
    
    # Confusion Matrix for KNN Classifier
    cm = confusion_matrix(y_test_knn, y_pred_knn, labels=['low', 'medium', 'high'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low', 'medium', 'high'])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix (KNN)")

    # Remove the empty plot (bottom-right)
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    plt.show()

# Split data for Regression and Classification
X_train, X_test, y_train_reg, y_test_reg = split_data(df_raw)  # For regression
X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_data(df_raw, classification=True)  # For classification

# 1. Run Linear Regression
y_test_lr, y_pred_lr = train_and_evaluate_linear_regression(X_train, X_test, y_train_reg, y_test_reg, apply_scaling=True)

# 2. Run K-Means Clustering
X_test_pca_kmeans, kmeans_labels = kmeans_clustering(X_train, X_test, df_raw.iloc[X_test.index], apply_scaling=True)

# 3. Run K-Nearest Neighbors Classifier
y_test_knn, y_pred_knn, accuracy_knn = train_and_evaluate_knn_classifier(X_train_cls, X_test_cls, y_train_cls, y_test_cls, apply_scaling=True)

# 4. Plot the results for Regression, Clustering, and Classification (KNN Confusion Matrix)
plot_results_combined(y_test_lr, y_pred_lr, X_test_pca_kmeans, kmeans_labels, y_test_knn, y_pred_knn)
