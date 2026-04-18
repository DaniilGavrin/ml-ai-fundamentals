# Оценка моделей машинного обучения

Правильная оценка моделей — ключевой этап в машинном обучении, определяющий качество и надежность ваших предсказаний.

## 1. Метрики для классификации

### Основные метрики
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy (точность) - доля правильных ответов
accuracy = accuracy_score(y_true, y_pred)

# Precision (точность положительного класса)
precision = precision_score(y_true, y_pred)

# Recall (полнота) - доля найденных положительных
recall = recall_score(y_true, y_pred)

# F1-score - гармоническое среднее precision и recall
f1 = f1_score(y_true, y_pred)
```

### Матрица ошибок (Confusion Matrix)
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Расшифровка:
# TN | FP
# FN | TP
```

### ROC-AUC и PR-AUC
```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

# ROC кривая
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Построение ROC кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ROC-AUC score
roc_auc = roc_auc_score(y_true, y_scores)

# Precision-Recall кривая (для несбалансированных данных)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)
```

### Логарифмические потери (Log Loss)
```python
from sklearn.metrics import log_loss

# Для бинарной классификации
logloss = log_loss(y_true, y_scores)

# Для многоклассовой
logloss = log_loss(y_true, y_scores_proba, labels=[0, 1, 2])
```

### Метрики для многоклассовой классификации
```python
from sklearn.metrics import classification_report

# Полный отчет по всем классам
print(classification_report(y_true, y_pred, digits=4))

# Средние значения
# macro - простое среднее по классам
# weighted - среднее с учетом поддержки классов
# micro - глобальный подсчет TP, FP, FN

precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
```

### Cohen's Kappa
```python
from sklearn.metrics import cohen_kappa_score

# Учет случайного угадывания
kappa = cohen_kappa_score(y_true, y_pred)
# Интерпретация:
# < 0 - хуже случайного
# 0-0.2 - слабое согласие
# 0.2-0.4 - умеренное
# 0.4-0.6 - хорошее
# 0.6-0.8 - очень хорошее
# 0.8-1 - почти идеальное
```

## 2. Метрики для регрессии

### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
# Интерпретация: средняя величина ошибки в единицах целевой переменной
```

### Mean Squared Error (MSE) и RMSE
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# MSE сильнее штрафует за большие ошибки
# RMSE интерпретируется в тех же единицах, что и y
```

### R² (Коэффициент детерминации)
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
# Интерпретация:
# 1 - идеальная модель
# 0 - модель не лучше константы (среднего)
# < 0 - модель хуже константы
```

### Mean Absolute Percentage Error (MAPE)
```python
def mape(y_true, y_pred):
    """Средняя абсолютная процентная ошибка"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_value = mape(y_true, y_pred)
# Интерпретация: средняя ошибка в процентах
```

### Symmetric MAPE (sMAPE)
```python
def smape(y_true, y_pred):
    """Симметричная MAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

smape_value = smape(y_true, y_pred)
```

## 3. Валидация моделей

### Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Для сохранения распределения классов
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
print(f"Scores: {scores}")
```

### Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### Leave-One-Out (LOO)
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
# Дорого вычислительно, но дает наименее смещенную оценку
```

### Time Series Split
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Важно: нет перемешивания, тест всегда после train
```

## 4. Анализ ошибок

### Learning Curves
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Train')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, test_mean, 'o-', label='Test')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
```

### Validation Curves
```python
from sklearn.model_selection import validation_curve

param_range = [1, 10, 50, 100, 200]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), X, y,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='accuracy', n_jobs=-1
)

# Анализ переобучения/недообучения
```

### Анализ ошибочных предсказаний
```python
# Найти примеры, где модель ошиблась
errors = y_pred != y_true
error_indices = np.where(errors)[0]

# Проанализировать характерные ошибки
for idx in error_indices[:10]:
    print(f"True: {y_true[idx]}, Predicted: {y_pred[idx]}")
    # Можно вывести сами данные или изображения
```

## 5. Сравнение моделей

### Статистические тесты
```python
from scipy import stats

# paired t-test для сравнения двух моделей
t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)

if p_value < 0.05:
    print("Модели статистически различаются")
else:
    print("Различия незначимы")
```

### McNemar's Test
```python
from statsmodels.stats.contingency_tables import mcnemar

# Таблица сопряженности ошибок
table = [[sum((p1==y) & (p2==y)), sum((p1==y) & (p2!=y))],
         [sum((p1!=y) & (p2==y)), sum((p1!=y) & (p2!=y))]]

result = mcnemar(table, exact=True)
print(f"p-value: {result.pvalue}")
```

## 6. Базовые модели (Baselines)

### Dummy Classifier
```python
from sklearn.dummy import DummyClassifier

# Стратегии:
# - most_frequent: всегда предсказывает самый частый класс
# - stratified: случайное предсказание по распределению
# - uniform: равномерное случайное предсказание
# - constant: всегда предсказывает заданный класс

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_score = dummy.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_score:.3f}")
```

### Dummy Regressor
```python
from sklearn.dummy import DummyRegressor

# Стратегии:
# - mean: всегда предсказывает среднее
# - median: всегда предсказывает медиану
# - quantile: предсказывает квантиль
# - constant: всегда предсказывает константу

dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
baseline_r2 = dummy.score(X_test, y_test)
print(f"Baseline R²: {baseline_r2:.3f}")
```

## Чек-лист оценки модели

- [ ] Выбрать подходящие метрики для задачи
- [ ] Использовать кросс-валидацию
- [ ] Сравнить с baseline моделью
- [ ] Проверить на переобучение (learning curves)
- [ ] Проанализировать ошибочные предсказания
- [ ] Проверить стабильность на разных фолдах
- [ ] Оценить калибровку вероятностей (если нужно)

## Рекомендации

1. **Всегда сравнивайте с baseline** — ваша модель должна быть лучше простого правила
2. **Используйте несколько метрик** — одна метрика может вводить в заблуждение
3. **Кросс-валидация обязательна** — особенно на маленьких датасетах
4. **Анализируйте ошибки** — это даст идеи для улучшения
5. **Учитывайте бизнес-контекст** — выбирайте метрики, важные для задачи
