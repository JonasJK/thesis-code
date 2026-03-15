#!/bin/bash
 
#SBATCH --job-name=install
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=1-00:30:00

#SBATCH --mem-per-cpu=30G
#SBATCH -G 1
cd /home/klugej/thesis/code/ || exit 1

module load CUDA/12.4.0
uv add --index https://pypi.org/simple --index https://pypi.nvidia.com "cuml-cu12==25.8.*"
uv sync
uv run python -c "import cuml.ensemble; print('cuML ensemble imported.')"

#!/bin/bash

# cuML vs sklearn Random Forest Benchmark Script

echo "Starting cuML vs sklearn Random Forest benchmark"
echo "=================================================="

SAMPLES=500000
FEATURES=500
INFORMATIVE_FEATURES=250
ESTIMATORS=100
MAX_DEPTH=15

echo "Dataset: ${SAMPLES} samples, ${FEATURES} features (${INFORMATIVE_FEATURES} informative)"
echo "Model: ${ESTIMATORS} estimators, max_depth=${MAX_DEPTH}"
echo ""

echo "Testing cuML Random Forest (GPU-accelerated)..."
uv run python -c "
import time
import cuml
from cuml.ensemble import RandomForestClassifier as cuMLRandomForest
import cuml.datasets.classification as cd
from cuml.model_selection import train_test_split
import numpy as np

print(f'cuML version: {cuml.__version__}')

print('Generating dataset...')
start_data = time.time()
X, y = cd.make_classification(
    n_samples=${SAMPLES}, 
    n_features=${FEATURES}, 
    n_informative=${INFORMATIVE_FEATURES}, 
    random_state=42
)
data_time = time.time() - start_data
print(f'Dataset generation time: {data_time:.3f}s')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples')

print('Training cuML RandomForest...')
start_train = time.time()
clf = cuMLRandomForest(
    n_estimators=${ESTIMATORS}, 
    max_depth=${MAX_DEPTH}, 
    random_state=42,
    n_streams=1  # Use single stream for consistent timing
)
clf.fit(X_train, y_train)
train_time = time.time() - start_train

print('Making predictions...')
start_pred = time.time()
predictions = clf.predict(X_test)
pred_time = time.time() - start_pred

start_score = time.time()
accuracy = clf.score(X_test, y_test)
score_time = time.time() - start_score

total_time = train_time + pred_time + score_time

print('')
print('cuML Results:')
print(f'  Training time:   {train_time:.3f}s')
print(f'  Prediction time: {pred_time:.3f}s')
print(f'  Scoring time:    {score_time:.3f}s')
print(f'  Total ML time:   {total_time:.3f}s')
print(f'  Accuracy:        {accuracy:.4f}')
print('')
"

echo "----------------------------------------"

echo "Testing sklearn Random Forest (CPU)..."
uv run python -c "
import time
import sklearn
from sklearn.ensemble import RandomForestClassifier as sklearnRandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

print(f'sklearn version: {sklearn.__version__}')

print('Generating dataset...')
start_data = time.time()
X, y = make_classification(
    n_samples=${SAMPLES}, 
    n_features=${FEATURES}, 
    n_informative=${INFORMATIVE_FEATURES}, 
    random_state=42
)
data_time = time.time() - start_data
print(f'Dataset generation time: {data_time:.3f}s')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples')

print('Training sklearn RandomForest...')
start_train = time.time()
clf = sklearnRandomForest(
    n_estimators=${ESTIMATORS}, 
    max_depth=${MAX_DEPTH}, 
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
clf.fit(X_train, y_train)
train_time = time.time() - start_train

print('Making predictions')
start_pred = time.time()
predictions = clf.predict(X_test)
pred_time = time.time() - start_pred

start_score = time.time()
accuracy = clf.score(X_test, y_test)
score_time = time.time() - start_score

total_time = train_time + pred_time + score_time

print('')
print('sklearn Results:')
print(f'  Training time:   {train_time:.3f}s')
print(f'  Prediction time: {pred_time:.3f}s')
print(f'  Scoring time:    {score_time:.3f}s')
print(f'  Total ML time:   {total_time:.3f}s')
print(f'  Accuracy:        {accuracy:.4f}')
print('')
"

echo "----------------------------------------"

echo "Running side-by-side comparison..."
uv run python -c "
import time
import numpy as np

print('Running cuML benchmark...')
start_cuml = time.time()

import cuml
from cuml.ensemble import RandomForestClassifier as cuMLRandomForest
import cuml.datasets.classification as cd
from cuml.model_selection import train_test_split as cuml_train_test_split

X, y = cd.make_classification(n_samples=${SAMPLES}, n_features=${FEATURES}, n_informative=${INFORMATIVE_FEATURES}, random_state=42)
X_train, X_test, y_train, y_test = cuml_train_test_split(X, y, test_size=0.2, random_state=42)

cuml_clf = cuMLRandomForest(n_estimators=${ESTIMATORS}, max_depth=${MAX_DEPTH}, random_state=42)
cuml_clf.fit(X_train, y_train)
cuml_acc = cuml_clf.score(X_test, y_test)
cuml_time = time.time() - start_cuml

print('Running sklearn benchmark...')
start_sklearn = time.time()

import sklearn
from sklearn.ensemble import RandomForestClassifier as sklearnRandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as sklearn_train_test_split

X, y = make_classification(n_samples=${SAMPLES}, n_features=${FEATURES}, n_informative=${INFORMATIVE_FEATURES}, random_state=42)
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)

sklearn_clf = sklearnRandomForest(n_estimators=${ESTIMATORS}, max_depth=${MAX_DEPTH}, random_state=42, n_jobs=-1)
sklearn_clf.fit(X_train, y_train)
sklearn_acc = sklearn_clf.score(X_test, y_test)
sklearn_time = time.time() - start_sklearn

print('')
print('FINAL COMPARISON')
print('===================')
print(f'cuML (GPU):     {cuml_time:.3f}s | Accuracy: {cuml_acc:.4f}')
print(f'sklearn (CPU):  {sklearn_time:.3f}s | Accuracy: {sklearn_acc:.4f}')
print('')

if cuml_time < sklearn_time:
    speedup = sklearn_time / cuml_time
    print(f'cuML is {speedup:.2f}x faster than sklearn.')
else:
    slowdown = cuml_time / sklearn_time  
    print(f'cuML is {slowdown:.2f}x slower than sklearn.')

acc_diff = abs(cuml_acc - sklearn_acc)
print(f'Accuracy difference: {acc_diff:.4f}')

if acc_diff < 0.01:
    print('Accuracy is essentially equivalent')
elif cuml_acc > sklearn_acc:
    print('cuML has better accuracy')
else:
    print('sklearn has better accuracy')
"

echo ""
echo "Benchmark complete."
