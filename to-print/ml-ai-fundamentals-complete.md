# Полная памятка по ML & AI

## 1. Основы машинного обучения

### Типы обучения
```python
# Обучение с учителем (Supervised Learning)
# Есть размеченные данные (X, y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Обучение без учителя (Unsupervised Learning)
# Только данные X без меток
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5)
model.fit(X)
labels = model.labels_

# Частичное обучение (Semi-supervised)
# Немного размеченных + много неразмеченных данных
from sklearn.semi_supervised import SelfTrainingClassifier
base_model = LogisticRegression()
model = SelfTrainingClassifier(base_model)
model.fit(X_partial_labeled, y_partial)

# Обучение с подкреплением (Reinforcement Learning)
# Агент учится через взаимодействие со средой
import gym
env = gym.make('CartPole-v1')
# agent learns through trial and error with rewards
```

### Bias-Variance Tradeoff
```python
# Высокий Bias (недообучение)
# Модель слишком простая, не улавливает закономерности
from sklearn.linear_model import LinearRegression
model = LinearRegression()  # Может дать высокий bias на сложных данных

# Высокая Variance (переобучение)
# Модель слишком сложная, запоминает шум
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=None)  # Склонна к переобучению

# Баланс
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=10, n_estimators=100)
# Умеренная сложность, регуляризация через ансамбль
```

### Кросс-валидация
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Stratified K-Fold (для несбалансированных классов)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='f1')

# Leave-One-Out (LOOCV)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

# Time Series Split (для временных рядов)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

---

## 2. Подготовка данных

### Обработка пропусков
```python
import pandas as pd
import numpy as np

# Удаление пропусков
df.dropna()  # Удалить строки с пропусками
df.dropna(axis=1)  # Удалить столбцы с пропусками
df.dropna(thresh=5)  # Оставить строки с минимум 5 не-NA значениями

# Заполнение константой
df.fillna(0)
df.fillna('Unknown')

# Заполнение статистиками
df['age'].fillna(df['age'].mean(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Интерполяция
df.interpolate(method='linear', inplace=True)
df.interpolate(method='polynomial', order=2, inplace=True)

# Forward/Backward fill (для временных рядов)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# KNN Imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Iterative Imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)
```

### Масштабирование признаков
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# StandardScaler (Z-score normalization)
# Приводит к mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Хорошо для: линейных моделей, SVM, нейросетей

# MinMaxScaler
# Масштабирует к диапазону [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
# Хорошо для: нейросетей, когда важны границы

# RobustScaler
# Использует медиану и квартили, устойчив к выбросам
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Хорошо для: данных с выбросами

# Normalizer
# Нормализует каждую строку к единичной норме
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)
# Хорошо для: текстовых данных, clustering

# Важно: fit на train, transform на train и test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Не fit_transform!
```

### Кодирование категориальных признаков
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd

# Label Encoding (для ординальных признаков)
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
# {'red': 2, 'green': 1, 'blue': 0}

# One-Hot Encoding (для номинальных признаков)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(df[['color']])
# red: [1,0,0], green: [0,1,0], blue: [0,0,1]

# Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
# drop_first=True избегает мультиколлинеарности

# Ordinal Encoding (когда есть порядок)
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['level_encoded'] = oe.fit_transform(df[['level']])

# Target Encoding (для high-cardinality признаков)
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['city'])
df_encoded = te.fit_transform(df, df['target'])
```

---

## 3. Инженерия признаков

### Создание признаков
```python
import pandas as pd
import numpy as np

# Полиномиальные признаки
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Взаимодействие признаков
df['income_per_person'] = df['income'] / df['family_size']
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                         labels=['child', 'young', 'adult', 'senior'])

# Датасет времени
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = pd.to_datetime(df['timestamp']).dt.month
df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                 3: 'spring', 4: 'spring', 5: 'spring',
                                 6: 'summer', 7: 'summer', 8: 'summer',
                                 9: 'fall', 10: 'fall', 11: 'fall'})

# Текстовые признаки
df['text_length'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()
df['avg_word_length'] = df['review'].apply(lambda x: np.mean([len(w) for w in x.split()]))

# Агрегации для групп
df['avg_income_by_city'] = df.groupby('city')['income'].transform('mean')
df['income_rank_in_city'] = df.groupby('city')['income'].rank(pct=True)
```

### Отбор признаков
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Фильтры (Filter methods)
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Mutual Information (нелинейные зависимости)
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Рекурсивное исключение (Wrapper methods)
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
X_selected = rfe.fit_transform(X, y)

# Важность признаков (Embedded methods)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 
                                   'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# L1 регуляризация для отбора
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# Корреляционный анализ
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_reduced = df.drop(to_drop, axis=1)
```

---

## 4. Оценка моделей

### Метрики классификации
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

# Основные метрики
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
# [[TN, FP],
#  [FN, TP]]

# ROC-AUC (для бинарной классификации)
roc_auc = roc_auc_score(y_true, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

# Precision-Recall Curve (для несбалансированных данных)
average_precision = average_precision_score(y_true, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

# Classification Report
print(classification_report(y_true, y_pred, digits=4))

# Для мультиклассовой классификации
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_micro = precision_score(y_true, y_pred, average='micro')
```

### Метрики регрессии
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import median_absolute_error, explained_variance_score
import numpy as np

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)

# R² Score (коэффициент детерминации)
r2 = r2_score(y_true, y_pred)
# 1.0 - идеальная модель, 0.0 - как mean baseline, <0 - хуже mean

# Median Absolute Error (устойчив к выбросам)
medae = median_absolute_error(y_true, y_pred)

# Explained Variance Score
evs = explained_variance_score(y_true, y_pred)

# Сравнение метрик
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
```

### Кросс-валидация стратегии
```python
from sklearn.model_selection import cross_validate, StratifiedKFold, GroupKFold
from sklearn.model_selection import TimeSeriesSplit, LeaveOneGroupOut

# Multiple scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    print(f"{metric}: {scores[f'test_{metric}'].mean():.3f} (+/- {scores[f'test_{metric}'].std():.3f})")

# Stratified K-Fold (сохраняет распределение классов)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group K-Fold (когда есть группы зависимых наблюдений)
gkfold = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkfold.split(X, y, groups=group_ids))

# Time Series Split (для временных рядов)
tscv = TimeSeriesSplit(n_splits=5)
# Train: [0..k], Test: [k+1..m] - нет утечки будущего

# Nested Cross-Validation (для честной оценки hyperparameter tuning)
from sklearn.model_selection import GridSearchCV, cross_val_score
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
```

---

## 5. Линейные модели

### Линейная регрессия
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import numpy as np

# Обычная линейная регрессия (OLS)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Ridge Regression (L2 регуляризация)
# Штрафует большие коэффициенты, уменьшает variance
ridge = Ridge(alpha=1.0)  # alpha = λ (сила регуляризации)
ridge.fit(X_train, y_train)

# Lasso Regression (L1 регуляризация)
# Обнуляет некоторые коэффициенты - feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
selected_features = np.where(lasso.coef_ != 0)[0]

# Elastic Net (комбинация L1 + L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: 1=Lasso, 0=Ridge
elastic.fit(X_train, y_train)

# Подбор alpha через кросс-валидацию
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

### Логистическая регрессия
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Бинарная классификация
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
# C = 1/λ (обратная сила регуляризации: маленькое C = сильная регуляризация)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Мультиклассовая классификация
# one-vs-rest (ovr): один бинарный классификатор на класс
model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')

# multinomial: настоящая мультиклассовая логистическая регрессия
model_multinomial = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Классы不平衡
model_weighted = LogisticRegression(class_weight='balanced')
# или вручную
model_custom_weight = LogisticRegression(class_weight={0: 1, 1: 10})

# Вероятности и пороги
y_proba = model.predict_proba(X_test)
threshold = 0.3  # Вместо стандартного 0.5
y_pred_custom = (y_proba[:, 1] >= threshold).astype(int)
```

---

## 6. Деревья решений и ансамбли

### Дерево решений
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Классификация
clf = DecisionTreeClassifier(
    criterion='gini',  # или 'entropy', 'log_loss'
    max_depth=5,       # максимальная глубина (регуляризация)
    min_samples_split=10,  # мин. образцов для разделения узла
    min_samples_leaf=5,    # мин. образцов в листе
    max_features='sqrt',   # число признаков для рассмотрения
    random_state=42
)
clf.fit(X_train, y_train)

# Регрессия
reg = DecisionTreeRegressor(
    criterion='squared_error',  # или 'absolute_error', 'poisson'
    max_depth=5,
    random_state=42
)
reg.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# Важность признаков
importances = clf.feature_importances_
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Классификация
rf_clf = RandomForestClassifier(
    n_estimators=100,      # число деревьев
    max_depth=10,          # глубина каждого дерева
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',   # sqrt для классификации, n_features/3 для регрессии
    bootstrap=True,        # бутстрэп выборки
    oob_score=True,        # out-of-bag оценка
    n_jobs=-1,             # параллелизация
    random_state=42
)
rf_clf.fit(X_train, y_train)
oob_score = rf_clf.oob_score_  # Оценка качества без отдельной валидации

# Регрессия
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
rf_reg.fit(X_train, y_train)

# Важность признаков
importances = rf_clf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
```

### Градиентный бустинг
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting (sklearn)
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,     # шаг градиентного спуска (0.01-0.1 обычно)
    max_depth=3,           # глубина деревьев (неглубокие деревья)
    min_samples_split=2,
    subsample=0.8,         # доля выборки для каждого дерева (стохастический GB)
    random_state=42
)
gb_clf.fit(X_train, y_train)

# Early stopping
from sklearn.model_selection import train_test_split
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2)

gb_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
gb_clf.fit(X_train_sub, y_train_sub, 
           eval_set=[(X_val, y_val)], 
           early_stopping_rounds=50,
           verbose=True)
best_n_estimators = gb_clf.best_iteration_

# XGBoost (более быстрая и продвинутая реализация)
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,  # аналог max_features
    reg_alpha=0,           # L1 регуляризация
    reg_lambda=1,          # L2 регуляризация
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# CatBoost (работает с категориальными признаками из коробки)
import catboost as cb
catboost_clf = cb.CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    cat_features=categorical_features,  # индексы категориальных признаков
    verbose=50,
    random_state=42,
    early_stopping_rounds=50
)
catboost_clf.fit(X_train, y_train, eval_set=(X_val, y_val))

# LightGBM (очень быстрый, good для больших данных)
import lightgbm as lgb
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,         # вместо max_depth (leaf-wise рост)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)
lgb_clf.fit(X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            eval_metric='binary_logloss',
            early_stopping_rounds=50)
```

---

## 7. Методы кластеризации

### K-Means
```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# K-Means
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',    # умная инициализация центроидов
    n_init=10,           # число запусков с разными центроидами
    max_iter=300,
    tol=1e-4,
    random_state=42
)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Выбор оптимального K (Elbow method)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Silhouette analysis
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.show()

# Метрики качества кластеризации
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)

# MiniBatch K-Means (для больших данных)
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
mb_kmeans.fit(X)
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# DBSCAN (Density-Based Spatial Clustering)
dbscan = DBSCAN(
    eps=0.5,           # радиус окрестности
    min_samples=5,     # мин. точек для dense региона
    metric='euclidean',
    n_jobs=-1
)
labels = dbscan.fit_predict(X)
# label = -1 означает шум (outlier)

# Подбор eps через k-distance graph
k = 5  # min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)
k_distances = distances[:, -1]  # k-е расстояние для каждой точки
k_distances_sorted = np.sort(k_distances)[::-1]

plt.plot(k_distances_sorted)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.axhline(y=0.5, color='r', linestyle='--', label='eps=0.5')
plt.legend()
plt.show()

# DBSCAN хорошо находит кластеры произвольной формы и выбросы
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')
```

### Иерархическая кластеризация
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    affinity='euclidean',   # или 'manhattan', 'cosine'
    linkage='ward'          # или 'complete', 'average', 'single'
)
labels = hc.fit_predict(X)

# Dendrogram (для визуализации иерархии)
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=15, color='r', linestyle='--')  # порог для выбора кластеров
plt.show()

# Разные методы linkage
# ward: минимизирует дисперсию внутри кластеров (good для сферических)
# complete: максимальное расстояние между точками кластеров
# average: среднее расстояние
# single: минимальное расстояние (склонно к chaining эффекту)
```

---

## 8. Снижение размерности

### PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"PC1 explains {explained_var[0]:.2%} of variance")
print(f"PC2 explains {explained_var[1]:.2%} of variance")
print(f"Cumulative: {cumulative_var[0]+cumulative_var[1]:.2%}")

# Выбор числа компонент
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.show()

# PCA с сохранением 95% дисперсии
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"Number of components: {pca_95.n_components_}")

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({explained_var[0]:.2%})')
plt.ylabel(f'PC2 ({explained_var[1]:.2%})')
plt.colorbar()
plt.title('PCA Projection')
plt.show()
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE (только для визуализации, не для preprocessing!)
tsne = TSNE(
    n_components=2,
    perplexity=30,       # баланс локальной/глобальной структуры (5-50)
    learning_rate=200,   # или 'auto'
    n_iter=1000,
    random_state=42,
    init='pca'           # или 'random'
)
X_tsne = tsne.fit_transform(X)

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar()
plt.title('t-SNE Projection')
plt.show()

# Подбор perplexity
perplexities = [5, 15, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
    axes[i].set_title(f'Perplexity = {perp}')
    axes[i].set_xlabel('t-SNE 1')
    axes[i].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

### UMAP (Uniform Manifold Approximation and Projection)
```python
import umap
import matplotlib.pyplot as plt

# UMAP (быстрее t-SNE, сохраняет глобальную структуру лучше)
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # баланс локальной/глобальной структуры (5-50)
    min_dist=0.1,        # мин. расстояние между точками в embedding (0-0.99)
    metric='euclidean',  # или 'manhattan', 'cosine', 'correlation'
    random_state=42
)
X_umap = reducer.fit_transform(X)

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.title('UMAP Projection')
plt.show()

# UMAP для preprocessing (не только визуализации)
reducer_full = umap.UMAP(
    n_components=10,     # можно больше 2-3 для preprocessing
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap_reduced = reducer_full.fit_transform(X)

# Использование в пайплайне
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('umap', umap.UMAP(n_components=10, random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
```

---

## 9. Нейронные сети: основы

### Перцептрон
```python
import numpy as np

# Простейший перцептрон (бинарная классификация)
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = self.step_function(linear_output)
                
                update = self.lr * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
    
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.step_function(x) for x in linear_output])

# Использование
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)
```

### Многослойный перцептрон (MLP)
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# MLP Classifier
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # архитектура сети
    activation='relu',             # или 'logistic', 'tanh', 'identity'
    solver='adam',                 # или 'sgd', 'lbfgs'
    alpha=0.0001,                  # L2 регуляризация
    batch_size=32,
    learning_rate='adaptive',      # или 'constant', 'invscaling'
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
mlp_clf.fit(X_train, y_train)

# MLP Regressor
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp_reg.fit(X_train, y_train)

# Важные гиперпараметры
# hidden_layer_sizes: больше слоев = сложнее зависимости, но риск переобучения
# activation: ReLU обычно лучший выбор для скрытых слоев
# alpha: регуляризация для предотвращения переобучения
# learning_rate_init: слишком большой = нестабильность, слишком маленький = медленная сходимость
```

### Функции активации
```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Диапазон: (0, 1), проблема vanishing gradient

# Tanh
def tanh(x):
    return np.tanh(x)
# Диапазон: (-1, 1), zero-centered, но всё ещё vanishing gradient

# ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)
# Проблема: dying ReLU (градиент = 0 для x < 0)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
# Решает проблему dying ReLU

# ELU (Exponential Linear Unit)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Swish / SiLU
def swish(x):
    return x * sigmoid(x)

# GELU (Gaussian Error Linear Unit) - используется в Transformers
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Softmax (для выходного слоя мультиклассовой классификации)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # для численной стабильности
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Визуализация
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, elu(x), label='ELU')
plt.plot(x, swish(x), label='Swish')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

### Backpropagation
```python
import numpy as np

# Упрощенная реализация backpropagation для MLP
class SimpleMLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        # Инициализация весов (Xavier initialization)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # ReLU для скрытых слоев, softmax для последнего
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softmax(z)
            
            self.activations.append(current)
        
        return current
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradient (cross-entropy loss + softmax)
        delta = self.activations[-1] - y  # производная loss по z
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Градиенты по весам и смещениям
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            # Обновление весов
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            # Градиент для предыдущего слоя (если не первый)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = -np.mean(np.sum(y * np.log(output + 1e-10), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Использование
network = SimpleMLP(layer_sizes=[784, 128, 64, 10])
network.fit(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)
predictions = network.predict(X_test)
```

---

## 10. Глубокое обучение: архитектуры

### Сверточные нейронные сети (CNN)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Базовая CNN для классификации изображений
model = keras.Sequential([
    # Входной слой: 28x28x1 (grayscale изображения)
    layers.Input(shape=(28, 28, 1)),
    
    # Первый сверточный блок
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Второй сверточный блок
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Flatten и fully connected слои
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Выходной слой
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Обучение
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr]
)

# Transfer Learning с предобученной моделью
base_model = keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Заморозить базовую модель

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Рекуррентные нейронные сети (RNN, LSTM, GRU)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Простая RNN (редко используется на практике из-за vanishing gradient)
model_rnn = keras.Sequential([
    layers.SimpleRNN(64, input_shape=(timesteps, features)),
    layers.Dense(1, activation='sigmoid')
])

# LSTM (Long Short-Term Memory) - решает проблему vanishing gradient
model_lstm = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Bidirectional LSTM (обрабатывает последовательность в обоих направлениях)
model_bi_lstm = keras.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
                        input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# GRU (Gated Recurrent Unit) - упрощенная версия LSTM, быстрее обучается
model_gru = keras.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.GRU(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Seq2Seq модель (для машинного перевода, summarization)
encoder_inputs = keras.Input(shape=(None, input_dim))
encoder_lstm = layers.LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = keras.Input(shape=(None, output_dim))
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.Dense(output_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

seq2seq_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

### Transformers и Attention Mechanism
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Multi-Head Attention слой
attention_layer = layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    value_dim=64
)

# Transformer Encoder блок
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = attention_layer(x, x)
    attention_output = layers.Dropout(dropout)(attention_output)
    x = layers.Add()([attention_output, inputs])  # Residual connection
    
    # Feed Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(head_size)(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.Add()([ff_output, x])  # Residual connection
    
    return x

# Полный Transformer Encoder
inputs = keras.Input(shape=(seq_length, input_dim))
x = inputs

# Позиционное кодирование (упрощенное)
pos_encoding = layers.Embedding(input_dim=seq_length, output_dim=input_dim)(tf.range(seq_length))
x = x + pos_encoding

# Несколько encoder блоков
for _ in range(4):
    x = transformer_encoder(x, head_size=256, num_heads=8, ff_dim=1024, dropout=0.1)

# Global pooling и выход
x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

transformer_model = keras.Model(inputs, outputs)

# Использование предобученных Transformers (Hugging Face)
from transformers import TFAutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_model = TFAutoModel.from_pretrained("bert-base-uncased")

inputs = keras.Input(shape=(None,), dtype=tf.int32)
attention_mask = keras.Input(shape=(None,), dtype=tf.int32)

embeddings = base_model(inputs, attention_mask=attention_mask)[0]
x = layers.GlobalAveragePooling1D()(embeddings)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

bert_classifier = keras.Model([inputs, attention_mask], outputs)
```

### Autoencoders
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Basic Autoencoder для снижения размерности
input_dim = 784
encoding_dim = 32

input_img = keras.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Denoising Autoencoder (для удаления шума)
noisy_input = keras.Input(shape=(input_dim,))
x = layers.GaussianNoise(0.1)(noisy_input)  # Добавление шума
encoded = layers.Dense(128, activation='relu')(x)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

denoising_autoencoder = keras.Model(noisy_input, decoded)
denoising_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Convolutional Autoencoder для изображений
input_img = keras.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

conv_autoencoder = keras.Model(input_img, decoded)
conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## 11. Регуляризация и оптимизация

### Методы регуляризации
```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# L1/L2 регуляризация весов
model = keras.Sequential([
    layers.Dense(128, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l1(0.01)),
    layers.Dropout(0.5),  # Dropout регуляризация
    layers.Dense(64, activation='relu',
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Dense(10, activation='softmax')
])

# Activity Regularization (штраф за большую активацию)
model.add(layers.Dense(64, activation='relu',
                      activity_regularizer=regularizers.l1(0.001)))

# Batch Normalization (также работает как регуляризатор)
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),  # Нормализация активаций
    layers.MaxPooling2D(),
    layers.Dropout(0.25)
])

# Data Augmentation (для изображений)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    # ... остальные слои
])

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=0.001
)

# Learning Rate Scheduling
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Оптимизаторы
```python
from tensorflow import keras

# SGD (Stochastic Gradient Descent)
optimizer_sgd = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov momentum ускоряет сходимость
)

# RMSprop (good для RNN)
optimizer_rmsprop = keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    epsilon=1e-07
)

# Adam (AdamW для лучшей регуляризации)
optimizer_adam = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False
)

# AdamW (Adam с decoupled weight decay)
optimizer_adamw = keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.01,  # Явный weight decay
    beta_1=0.9,
    beta_2=0.999
)

# Выбор оптимизатора
# SGD + Momentum: хорош для обобщения, но требует тщательной настройки LR
# Adam: быстро сходится, good по умолчанию, но может generalize хуже
# AdamW: лучшая регуляризация чем Adam, особенно для Transformers
# RMSprop: традиционный выбор для RNN
```

---

## 12. Ансамбли и Hyperparameter Tuning

### Voting и Stacking
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Hard Voting (голосование большинством)
voting_hard = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='hard'
)

# Soft Voting (усреднение вероятностей)
voting_soft = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    voting='soft'
)

# Stacking (мета-обучение)
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(n_estimators=10)),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'  # или 'predict', 'decision_function'
)

# Bagging (Bootstrap Aggregating)
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)
```

### Grid Search и Random Search
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# Grid Search (полный перебор)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Random Search (случайный поиск, эффективнее для большого пространства)
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # число итераций
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")

# Bayesian Optimization (Optuna)
import optuna
from optuna.integration import SKLearnPruningCallback

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    clf = RandomForestClassifier(**params)
    
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")
```

---

## 13. Этичность и безопасность ML

### Fairness Metrics
```python
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# Демографический паритет (равные положительные ставки)
def demographic_parity(y_true, y_pred, sensitive_feature):
    groups = np.unique(sensitive_feature)
    positive_rates = {}
    
    for group in groups:
        mask = sensitive_feature == group
        positive_rate = np.mean(y_pred[mask] == 1)
        positive_rates[group] = positive_rate
    
    parity_diff = max(positive_rates.values()) - min(positive_rates.values())
    return positive_rates, parity_diff

# Equal Opportunity (равные TPR)
def equal_opportunity(y_true, y_pred, sensitive_feature):
    groups = np.unique(sensitive_feature)
    tpr_rates = {}
    
    for group in groups:
        mask = sensitive_feature == group
        actual_positive = y_true[mask] == 1
        if np.sum(actual_positive) > 0:
            tpr = np.sum((y_pred[mask] == 1) & actual_positive) / np.sum(actual_positive)
            tpr_rates[group] = tpr
    
    opportunity_diff = max(tpr_rates.values()) - min(tpr_rates.values())
    return tpr_rates, opportunity_diff

# Predictive Parity (равная положительная прогностическая ценность)
def predictive_parity(y_true, y_pred, sensitive_feature):
    groups = np.unique(sensitive_feature)
    ppv_rates = {}
    
    for group in groups:
        mask = sensitive_feature == group
        predicted_positive = y_pred[mask] == 1
        if np.sum(predicted_positive) > 0:
            ppv = np.sum((y_true[mask] == 1) & predicted_positive) / np.sum(predicted_positive)
            ppv_rates[group] = ppv
    
    return ppv_rates

# Использование
positive_rates, parity_diff = demographic_parity(y_true, y_pred, sensitive_feature)
print(f"Demographic Parity Difference: {parity_diff:.3f}")

tpr_rates, opp_diff = equal_opportunity(y_true, y_pred, sensitive_feature)
print(f"Equal Opportunity Difference: {opp_diff:.3f}")
```

### Interpretability (SHAP, LIME)
```python
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

# SHAP (SHapley Additive exPlanations)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# TreeExplainer (специально для деревьев, быстрый)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (важность признаков)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Dependence plot (как признак влияет на предсказание)
shap.dependence_plot("age", shap_values[1], X_test, feature_names=feature_names)

# Force plot для одного примера
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], 
                X_test.iloc[0,:], feature_names=feature_names)

# LIME (Local Interpretable Model-agnostic Explanations)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Объяснение для одного примера
explanation = explainer_lime.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=5
)

explanation.show_in_notebook()
explanation.as_pyplot_figure()
```

### Adversarial Attacks и защита
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# FGSM (Fast Gradient Sign Method) атака
def fgsm_attack(model, image, label, epsilon=0.01):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.Variable(image)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = keras.losses.sparse_categorical_crossentropy([label], prediction)
    
    gradients = tape.gradient(loss, image)
    signed_grad = tf.sign(gradients)
    adversarial_image = image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

# PGD (Projected Gradient Descent) атака
def pgd_attack(model, image, label, epsilon=0.1, num_steps=10, step_size=0.01):
    adversarial_image = image.copy()
    
    for _ in range(num_steps):
        adversarial_image = fgsm_attack(model, adversarial_image, label, step_size)
        # Проекция обратно в epsilon-шар
        diff = adversarial_image - image
        diff = np.clip(diff, -epsilon, epsilon)
        adversarial_image = np.clip(image + diff, 0, 1)
    
    return adversarial_image

# Adversarial Training (защита)
def adversarial_training(model, X_train, y_train, epochs=10, epsilon=0.01):
    for epoch in range(epochs):
        adversarial_examples = []
        
        for i in range(len(X_train)):
            adv_image = fgsm_attack(model, X_train[i:i+1], y_train[i], epsilon)
            adversarial_examples.append(adv_image)
        
        adversarial_examples = np.vstack(adversarial_examples)
        combined_X = np.vstack([X_train, adversarial_examples])
        combined_y = np.concatenate([y_train, y_train])
        
        model.fit(combined_X, combined_y, epochs=1, verbose=0)
    
    return model

# Проверка robustness
original_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
adversarial_images = np.array([fgsm_attack(model, img, label, 0.01) 
                               for img, label in zip(X_test[:100], y_test[:100])])
adversarial_accuracy = model.evaluate(adversarial_images, y_test[:100], verbose=0)[1]

print(f"Original accuracy: {original_accuracy:.3f}")
print(f"Adversarial accuracy: {adversarial_accuracy:.3f}")
```

### Differential Privacy
```python
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
import tensorflow as tf

# Differential Privacy SGD
optimizer = DPGradientDescentGaussianOptimizer(
    l2_norm_clip=1.0,           # clipping norm для градиентов
    noise_multiplier=0.5,       # уровень шума
    num_microbatches=32,        # число микро-батчей
    learning_rate=0.01
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Расчет privacy budget (epsilon)
compute_dp_sgd_privacy(
    n=60000,                    # размер датасета
    batch_size=256,
    noise_multiplier=0.5,
    epochs=10,
    delta=1e-5                  # вероятность failure
)

# Federated Learning (децентрализованное обучение)
import tensorflow_federated as tff

def create_federated_model():
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    return model

def model_fn():
    keras_model = create_federated_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_data.element_spec,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

federated_algorithm = tff.learning.build_federated_averaging_process(model_fn)

# Федеративное обучение на клиентах
state = federated_algorithm.initialize()
for round_num in range(10):
    state, metrics = federated_algorithm.next(state, federated_train_data)
    print(f"Round {round_num}: {metrics}")
```

---

*Полная памятка по ML & AI Fundamentals v1.0.0*
