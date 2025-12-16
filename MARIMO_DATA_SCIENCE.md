# Marimo for Data Science, ML, and AI Development

A comprehensive guide to using Marimo for data science, data analysis, machine learning, and AI development.

## Table of Contents

1. [Overview](#overview)
2. [Data Science with Marimo](#data-science-with-marimo)
3. [Data Analysis Workflows](#data-analysis-workflows)
4. [Machine Learning Development](#machine-learning-development)
5. [AI Development](#ai-development)
6. [Complete Workflow Examples](#complete-workflow-examples)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Integration with ML/AI Tools](#integration-with-mlai-tools)

---

## Overview

Marimo is an excellent tool for data science, ML, and AI development because of its:

- **Reactive execution**: Automatically updates when data or parameters change
- **Reproducibility**: Deterministic execution order
- **Interactive UI**: Built-in components for parameter tuning
- **Type safety**: Better error detection in data pipelines
- **Version control**: Python files instead of JSON notebooks

### Why Marimo for Data Science?

✅ **Automatic dependency tracking** - No manual re-running of cells  
✅ **Interactive parameter tuning** - UI sliders and inputs for hyperparameters  
✅ **Reproducible experiments** - Same inputs always produce same outputs  
✅ **Easy sharing** - Export to HTML or Python scripts  
✅ **Type checking** - Catch errors early in data pipelines  

---

## Data Science with Marimo

### 1. Data Loading and Exploration

```python
import marimo

__all__ = ["df", "summary", "missing_data", "data_types"]

# Cell 1: Load data
import pandas as pd
import numpy as np

df = pd.read_csv("sales_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Cell 2: Basic statistics
summary = df.describe()
print("Summary Statistics:")
print(summary)

# Cell 3: Data quality check
missing_data = df.isnull().sum()
print("Missing values per column:")
print(missing_data[missing_data > 0])

# Cell 4: Data types
data_types = df.dtypes
print("Data types:")
print(data_types)
```

### 2. Interactive Data Exploration

```python
import marimo
import marimo.ui as mui
import pandas as pd
import plotly.express as px

__all__ = ["selected_column", "filtered_df", "plot"]

# Cell 1: UI for column selection
selected_column = mui.dropdown(
    options=list(df.columns),
    value=df.columns[0],
    label="Select Column to Analyze"
)

# Cell 2: Filter data based on selection
filtered_df = df[[selected_column]].copy()

# Cell 3: Interactive visualization
plot = px.histogram(
    filtered_df,
    x=selected_column,
    title=f"Distribution of {selected_column}",
    nbins=30
)
```

### 3. Data Cleaning Pipeline

```python
import marimo

__all__ = ["raw_df", "cleaned_df", "cleaning_report"]

# Cell 1: Load raw data
raw_df = pd.read_csv("raw_data.csv")
print(f"Raw data shape: {raw_df.shape}")

# Cell 2: Remove duplicates
cleaned_df = raw_df.drop_duplicates()
print(f"After removing duplicates: {cleaned_df.shape}")

# Cell 3: Handle missing values
# Strategy: Fill numeric with median, categorical with mode
numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
categorical_cols = cleaned_df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)

for col in categorical_cols:
    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)

# Cell 4: Generate cleaning report
cleaning_report = {
    "original_rows": len(raw_df),
    "final_rows": len(cleaned_df),
    "duplicates_removed": len(raw_df) - len(cleaned_df),
    "missing_values_filled": raw_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
}
print("Cleaning Report:", cleaning_report)
```

### 4. Feature Engineering

```python
import marimo

__all__ = ["df", "engineered_df", "feature_list"]

# Cell 1: Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Cell 2: Create new features
engineered_df = df.copy()

# Date features
if 'date' in engineered_df.columns:
    engineered_df['date'] = pd.to_datetime(engineered_df['date'])
    engineered_df['year'] = engineered_df['date'].dt.year
    engineered_df['month'] = engineered_df['date'].dt.month
    engineered_df['day_of_week'] = engineered_df['date'].dt.dayofweek

# Numerical features
if 'price' in engineered_df.columns and 'quantity' in engineered_df.columns:
    engineered_df['total_value'] = engineered_df['price'] * engineered_df['quantity']

# Categorical encoding
if 'category' in engineered_df.columns:
    engineered_df = pd.get_dummies(engineered_df, columns=['category'], prefix='cat')

# Cell 3: List all features
feature_list = list(engineered_df.columns)
print(f"Total features: {len(feature_list)}")
print("Features:", feature_list)
```

---

## Data Analysis Workflows

### 1. Statistical Analysis

```python
import marimo
import pandas as pd
import numpy as np
from scipy import stats

__all__ = ["df", "statistical_tests", "correlation_matrix"]

# Cell 1: Load data
df = pd.read_csv("experiment_data.csv")

# Cell 2: Descriptive statistics
descriptive_stats = {
    "mean": df.select_dtypes(include=[np.number]).mean(),
    "median": df.select_dtypes(include=[np.number]).median(),
    "std": df.select_dtypes(include=[np.number]).std(),
    "skewness": df.select_dtypes(include=[np.number]).skew(),
    "kurtosis": df.select_dtypes(include=[np.number]).kurtosis()
}

# Cell 3: Statistical tests
# T-test for comparing two groups
if 'group' in df.columns and 'value' in df.columns:
    group_a = df[df['group'] == 'A']['value']
    group_b = df[df['group'] == 'B']['value']
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    statistical_tests = {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    print("T-test results:", statistical_tests)

# Cell 4: Correlation analysis
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print("Correlation Matrix:")
print(correlation_matrix)
```

### 2. Time Series Analysis

```python
import marimo
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

__all__ = ["ts_data", "decomposed", "forecast"]

# Cell 1: Load time series data
ts_data = pd.read_csv("time_series.csv", parse_dates=['date'], index_col='date')
ts_data = ts_data['value'].resample('D').mean()  # Daily aggregation

# Cell 2: Decompose time series
decomposed = seasonal_decompose(ts_data, model='additive', period=365)

# Cell 3: Simple forecasting (moving average)
window = 30
forecast = ts_data.rolling(window=window).mean()
print(f"Forecast for next period: {forecast.iloc[-1]:.2f}")

# Cell 4: Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))
fig.update_layout(title="Time Series Analysis")
plot = fig
```

### 3. A/B Testing Analysis

```python
import marimo
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

__all__ = ["ab_data", "test_results", "visualization"]

# Cell 1: Load A/B test data
ab_data = pd.read_csv("ab_test_results.csv")
print(f"Total participants: {len(ab_data)}")
print(f"Group A: {len(ab_data[ab_data['group'] == 'A'])}")
print(f"Group B: {len(ab_data[ab_data['group'] == 'B'])}")

# Cell 2: Calculate metrics
group_a_metrics = ab_data[ab_data['group'] == 'A']['conversion'].agg({
    'mean': 'mean',
    'std': 'std',
    'count': 'count'
})

group_b_metrics = ab_data[ab_data['group'] == 'B']['conversion'].agg({
    'mean': 'mean',
    'std': 'std',
    'count': 'count'
})

# Cell 3: Statistical significance test
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(ab_data['group'], ab_data['converted'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

test_results = {
    "chi_squared": chi2,
    "p_value": p_value,
    "significant": p_value < 0.05,
    "group_a_conversion": group_a_metrics['mean'],
    "group_b_conversion": group_b_metrics['mean'],
    "lift": (group_b_metrics['mean'] - group_a_metrics['mean']) / group_a_metrics['mean'] * 100
}

print("A/B Test Results:", test_results)

# Cell 4: Visualization
visualization = px.bar(
    x=['Group A', 'Group B'],
    y=[group_a_metrics['mean'], group_b_metrics['mean']],
    title="Conversion Rates by Group",
    labels={'y': 'Conversion Rate', 'x': 'Group'}
)
```

---

## Machine Learning Development

### 1. ML Model Training Pipeline

```python
import marimo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

__all__ = ["X_train", "X_test", "y_train", "y_test", "model", "predictions", "metrics"]

# Cell 1: Load and prepare data
df = pd.read_csv("ml_dataset.csv")
X = df.drop('target', axis=1)
y = df['target']

# Cell 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Cell 3: Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Cell 4: Make predictions
predictions = model.predict(X_test)

# Cell 5: Evaluate model
accuracy = accuracy_score(y_test, predictions)
metrics = {
    "accuracy": accuracy,
    "classification_report": classification_report(y_test, predictions),
    "confusion_matrix": confusion_matrix(y_test, predictions)
}
print(f"Accuracy: {accuracy:.4f}")
```

### 2. Hyperparameter Tuning with UI

```python
import marimo
import marimo.ui as mui
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

__all__ = ["n_estimators", "max_depth", "min_samples_split", "tuned_model", "tuned_accuracy"]

# Cell 1: UI controls for hyperparameters
n_estimators = mui.slider(
    min=10,
    max=200,
    value=100,
    step=10,
    label="Number of Estimators"
)

max_depth = mui.slider(
    min=3,
    max=20,
    value=10,
    step=1,
    label="Max Depth"
)

min_samples_split = mui.slider(
    min=2,
    max=20,
    value=2,
    step=1,
    label="Min Samples Split"
)

# Cell 2: Train model with selected hyperparameters
tuned_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
tuned_model.fit(X_train, y_train)

# Cell 3: Evaluate tuned model
tuned_predictions = tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_predictions)
print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")
```

### 3. Model Comparison

```python
import marimo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

__all__ = ["models", "results"]

# Cell 1: Define models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Cell 2: Train and evaluate all models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1_score": f1_score(y_test, predictions, average='weighted')
    }

# Cell 3: Display results
results_df = pd.DataFrame(results).T
print("Model Comparison:")
print(results_df)
```

### 4. Cross-Validation

```python
import marimo
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

__all__ = ["cv_scores", "cv_results"]

# Cell 1: Setup cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cell 2: Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Cell 3: Analyze results
cv_results = {
    "mean_accuracy": cv_scores.mean(),
    "std_accuracy": cv_scores.std(),
    "min_accuracy": cv_scores.min(),
    "max_accuracy": cv_scores.max(),
    "all_scores": cv_scores.tolist()
}
print("Cross-Validation Results:")
print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
```

### 5. Feature Importance Analysis

```python
import marimo
import pandas as pd
import plotly.express as px

__all__ = ["feature_importance", "importance_plot"]

# Cell 1: Get feature importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Cell 2: Visualize feature importance
importance_plot = px.bar(
    feature_importance.head(20),
    x='importance',
    y='feature',
    orientation='h',
    title="Top 20 Feature Importances"
)
```

---

## AI Development

### 1. Deep Learning Model Development

```python
import marimo
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

__all__ = ["model", "history", "evaluation"]

# Cell 1: Prepare data for neural network
# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cell 2: Build neural network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Cell 3: Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Cell 4: Evaluate model
evaluation = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {evaluation[1]:.4f}")
```

### 2. Natural Language Processing

```python
import marimo
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

__all__ = ["vectorizer", "X_train_tfidf", "nlp_model", "nlp_predictions"]

# Cell 1: Load text data
text_data = pd.read_csv("text_data.csv")
print(f"Text samples: {len(text_data)}")

# Cell 2: Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(text_data['text'])
y_train_text = text_data['label']

# Cell 3: Train NLP model
nlp_model = MultinomialNB()
nlp_model.fit(X_train_tfidf, y_train_text)

# Cell 4: Make predictions
test_text = vectorizer.transform(text_data['text'].iloc[:100])
nlp_predictions = nlp_model.predict(test_text)
print("NLP Predictions:", nlp_predictions[:10])
```

### 3. Computer Vision

```python
import marimo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__all__ = ["cv_model", "cv_history"]

# Cell 1: Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

# Cell 2: Build CNN model
cv_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

cv_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Cell 3: Train model
cv_history = cv_model.fit(
    datagen.flow(X_train_images, y_train_images, batch_size=32),
    epochs=20,
    validation_data=(X_val_images, y_val_images)
)
```

### 4. Reinforcement Learning Setup

```python
import marimo
import numpy as np
import gym

__all__ = ["env", "q_table", "training_results"]

# Cell 1: Initialize environment
env = gym.make('CartPole-v1')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Cell 2: Initialize Q-table
q_table = np.zeros((state_space, action_space))
print(f"Q-table shape: {q_table.shape}")

# Cell 3: Q-learning training
def train_q_learning(episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    training_results = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, _ = env.step(action)
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state
            total_reward += reward
        
        training_results.append(total_reward)
    
    return training_results

training_results = train_q_learning()
print(f"Average reward: {np.mean(training_results[-100:]):.2f}")
```

---

## Complete Workflow Examples

### Example 1: End-to-End ML Pipeline

```python
import marimo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

__all__ = ["final_model", "model_metrics", "saved_model_path"]

# Cell 1: Load data
df = pd.read_csv("customer_data.csv")
print(f"Dataset: {df.shape}")

# Cell 2: Data preprocessing
# Handle missing values
df = df.fillna(df.median())

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Cell 3: Feature selection
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cell 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cell 5: Train model
final_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
final_model.fit(X_train_scaled, y_train)

# Cell 6: Evaluate
predictions = final_model.predict(X_test_scaled)
model_metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "classification_report": classification_report(y_test, predictions)
}
print(f"Model Accuracy: {model_metrics['accuracy']:.4f}")

# Cell 7: Save model
saved_model_path = "models/churn_model.pkl"
joblib.dump(final_model, saved_model_path)
joblib.dump(scaler, "models/scaler.pkl")
print(f"Model saved to {saved_model_path}")
```

### Example 2: Interactive ML Experimentation

```python
import marimo
import marimo.ui as mui
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

__all__ = ["model_params", "trained_model", "performance_plot"]

# Cell 1: Interactive parameter selection
model_params = {
    "n_estimators": mui.slider(10, 500, value=100, step=10, label="Number of Trees"),
    "max_depth": mui.slider(3, 30, value=10, step=1, label="Max Depth"),
    "min_samples_split": mui.slider(2, 20, value=2, step=1, label="Min Samples Split"),
    "min_samples_leaf": mui.slider(1, 10, value=1, step=1, label="Min Samples Leaf")
}

# Cell 2: Train model with selected parameters
trained_model = RandomForestClassifier(
    n_estimators=model_params["n_estimators"],
    max_depth=model_params["max_depth"],
    min_samples_split=model_params["min_samples_split"],
    min_samples_leaf=model_params["min_samples_leaf"],
    random_state=42
)
trained_model.fit(X_train, y_train)

# Cell 3: Evaluate and visualize
predictions = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

performance_plot = px.bar(
    x=["Current Model"],
    y=[accuracy],
    title=f"Model Accuracy: {accuracy:.4f}",
    labels={'y': 'Accuracy', 'x': 'Model'}
)
```

---

## Best Practices

### 1. Organize Your Workflow

```python
# Group 1: Imports and Setup
import marimo
__all__ = ["..."]

# Group 2: Data Loading
# Load and initial exploration

# Group 3: Data Preprocessing
# Cleaning, transformation, feature engineering

# Group 4: Model Development
# Training, evaluation, tuning

# Group 5: Results and Visualization
# Metrics, plots, reports
```

### 2. Use Type Hints

```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data and return features and target."""
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y
```

### 3. Document Your Experiments

```python
# Cell: Experiment documentation
experiment_log = {
    "date": "2024-01-15",
    "model_type": "Random Forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "accuracy": 0.85,
    "notes": "First baseline model"
}
```

### 4. Version Control Your Data

```python
# Always specify data version
DATA_VERSION = "v1.2"
df = pd.read_csv(f"data/processed_data_{DATA_VERSION}.csv")
```

---

## Performance Optimization

### 1. Caching Expensive Operations

```python
import marimo
from functools import lru_cache

@lru_cache(maxsize=1)
def expensive_computation(data_hash: str) -> dict:
    # Expensive operation
    result = complex_processing()
    return result

# Use cached result
data_hash = hash(str(df.values.tobytes()))
result = expensive_computation(data_hash)
```

### 2. Parallel Processing

```python
from joblib import Parallel, delayed

def process_chunk(chunk):
    return process_data(chunk)

# Process in parallel
results = Parallel(n_jobs=4)(
    delayed(process_chunk)(chunk) 
    for chunk in data_chunks
)
```

### 3. Memory Management

```python
# Use appropriate data types
df['column'] = df['column'].astype('category')  # Saves memory
df = df.drop(columns=['unused_column'])  # Remove unused data
```

---

## Integration with ML/AI Tools

### 1. MLflow Integration

```python
import mlflow
import mlflow.sklearn

# Cell 1: Start MLflow run
mlflow.start_run()

# Cell 2: Log parameters and metrics
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", accuracy)

# Cell 3: Log model
mlflow.sklearn.log_model(model, "model")

# Cell 4: End run
mlflow.end_run()
```

### 2. Weights & Biases (W&B)

```python
import wandb

# Cell 1: Initialize W&B
wandb.init(project="marimo-ml-project")

# Cell 2: Log metrics
wandb.log({"accuracy": accuracy, "loss": loss})

# Cell 3: Log model
wandb.log_model(model, "model")
```

### 3. TensorBoard

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Cell 1: Setup TensorBoard
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Cell 2: Train with TensorBoard
model.fit(
    X_train, y_train,
    epochs=50,
    callbacks=[tensorboard_callback]
)
```

---

## Conclusion

Marimo is a powerful tool for data science, machine learning, and AI development because it:

1. **Automates dependency tracking** - No manual re-running
2. **Provides interactive UI** - Easy parameter tuning
3. **Ensures reproducibility** - Deterministic execution
4. **Supports all ML/AI workflows** - From data loading to model deployment
5. **Integrates with ML tools** - MLflow, W&B, TensorBoard

### Use Cases

✅ **Data Science**: Exploratory data analysis, statistical modeling  
✅ **Machine Learning**: Model training, hyperparameter tuning, evaluation  
✅ **Deep Learning**: Neural network development, computer vision, NLP  
✅ **AI Development**: Reinforcement learning, model experimentation  
✅ **Research**: Reproducible experiments, parameter sweeps  

### Next Steps

1. Start with data exploration workflows
2. Build ML pipelines with reactive updates
3. Use UI components for interactive experimentation
4. Integrate with your favorite ML tools
5. Share your notebooks as HTML or Python scripts

**Marimo makes data science and ML development more interactive, reproducible, and enjoyable! 🚀**

