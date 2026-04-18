# Инженерия признаков

Инженерия признаков (Feature Engineering) — процесс создания новых признаков из существующих данных для улучшения качества моделей машинного обучения.

## 1. Создание признаков

### Извлечение из дат и времени
```python
import pandas as pd

df['date'] = pd.to_datetime(df['timestamp'])

# Извлечение компонентов
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=понедельник
df['hour'] = df['date'].dt.hour
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Циклическое кодирование
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

### Извлечение из текста
```python
# Длина текста
df['text_length'] = df['text'].str.len()

# Количество слов
df['word_count'] = df['text'].str.split().str.len()

# Количество предложений
df['sentence_count'] = df['text'].str.count(r'[.!?]')

# Средняя длина слова
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x else 0
)

# Наличие заглавных букв
df['has_uppercase'] = df['text'].str.contains('[A-ZА-Я]').astype(int)

# TF-IDF векторизация
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['text'])
```

### Извлечение из чисел
```python
# Бининг (разбиение на интервалы)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 55, 100], 
                          labels=['young', 'adult', 'middle', 'senior'])

# Квантильное разбиение
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Логарифмическое преобразование
df['log_income'] = np.log1p(df['income'])

# Отношения и взаимодействия
df['debt_to_income'] = df['debt'] / df['income']
df['price_per_sqm'] = df['price'] / df['area']
```

### Агрегированные признаки
```python
# Групповые статистики
group_stats = df.groupby('category')['value'].agg(['mean', 'std', 'min', 'max', 'count'])
df = df.merge(group_stats, on='category', suffixes=('', '_cat'))

# Скользящие средние (для временных рядов)
df['moving_avg_7'] = df['value'].rolling(window=7).mean()
df['moving_avg_30'] = df['value'].rolling(window=30).mean()

# Лаговые признаки
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)

# Разности
df['diff_1'] = df['value'].diff(1)
df['diff_7'] = df['value'].diff(7)
```

## 2. Отбор признаков

### Фильтрация по дисперсии
```python
from sklearn.feature_selection import VarianceThreshold

# Удаление признаков с низкой дисперсией
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
```

### Корреляционный анализ
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Матрица корреляций
corr_matrix = df.corr()

# Визуализация
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()

# Удаление сильно коррелирующих признаков
corr_threshold = 0.95
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > corr_threshold)]
df_reduced = df.drop(columns=to_drop)
```

### Важность признаков (Feature Importance)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# На основе случайного леса
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Получение важности
importances = rf.feature_importances_
feature_names = X.columns

# Отбор по важности
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_selected = selector.transform(X)
selected_features = feature_names[selector.get_support()]
```

### Рекурсивное исключение признаков (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFE с логистической регрессией
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=10, step=1)
rfe.fit(X, y)

# Ранжированные признаки
ranking = rfe.ranking_  # 1 = выбрано, больше = отброшено раньше
selected_features = X.columns[rfe.support_]
```

### Статистические тесты
```python
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

# F-тест (ANOVA) для непрерывных признаков
f_scores, p_values = f_classif(X, y)

# Хи-квадрат для категориальных признаков
chi2_scores, p_values = chi2(X, y)

# Взаимная информация
mi_scores = mutual_info_classif(X, y, random_state=42)

# Отбор по p-value
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

## 3. Преобразование признаков

### Полиномиальные признаки
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)

# Только взаимодействия
poly_interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interaction = poly_interaction.fit_transform(X)
```

### Сплайны и базисные функции
```python
from patsy import dmatrix

# B-сплайны
spline_features = dmatrix("bs(x, df=5, degree=3)", {"x": df['feature']}, return_type='dataframe')

# Натуральные сплайны
natural_spline = dmatrix("cr(x, df=5)", {"x": df['feature']}, return_type='dataframe')
```

### Target Encoding (с регуляризацией)
```python
def smooth_target_encoding(df, column, target, alpha=10):
    """Target encoding с лапласовской регуляризацией"""
    global_mean = df[target].mean()
    
    agg = df.groupby(column)[target].agg(['mean', 'count'])
    smoothing = (alpha * global_mean + agg['count'] * agg['mean']) / (alpha + agg['count'])
    
    df[f'{column}_target_enc'] = df[column].map(smoothing)
    return df

df = smooth_target_encoding(df, 'category', 'target', alpha=20)
```

## 4. Обработка категориальных признаков

### Частотное кодирование
```python
# Кодирование частотой категории
freq_encoder = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq_encoder)
```

### Leave-One-Out Encoding
```python
def leave_one_out_encode(df, column, target):
    """LOO encoding для избежания data leakage"""
    loo_encoded = []
    for i in range(len(df)):
        mask = np.arange(len(df)) != i
        mean_target = df.loc[mask].groupby(column)[target].mean()
        loo_encoded.append(mean_target[df.iloc[i][column]])
    return loo_encoded

df['category_loo'] = leave_one_out_encode(df, 'category', 'target')
```

### Hashing Encoder
```python
from category_encoders import HashingEncoder

# Хэширование для высокоразмерных категориальных признаков
he = HashingEncoder(cols=['category1', 'category2'], n_components=16)
X_hashed = he.fit_transform(X)
```

## 5. Автоматизация инженерии признаков

### Featuretools
```python
import featuretools as ft

# Создание entity set
es = ft.EntitySet(id='transactions')

# Добавление таблиц
es.add_dataframe(dataframe=df_transactions, dataframe_name='transactions', index='id')
es.add_dataframe(dataframe=df_customers, dataframe_name='customers', index='customer_id')

# Определение связей
es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Глубокий синтез признаков
feature_matrix, features = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,
    agg_primitives=['mean', 'sum', 'std'],
    trans_primitives=['add', 'multiply']
)
```

### tsfresh для временных рядов
```python
from tsfresh import extract_features

# Автоматическое извлечение признаков из временных рядов
features = extract_features(timeseries_data, column_id='id', column_sort='time')
```

## Чек-лист инженерии признаков

- [ ] Создать признаки из дат/времени
- [ ] Извлечь признаки из текста (если есть)
- [ ] Создать агрегированные признаки
- [ ] Проверить корреляции
- [ ] Отобрать важные признаки
- [ ] Применить полиномиальные признаки (если нужно)
- [ ] Закодировать категориальные признаки
- [ ] Проверить на мультиколлинеарность

## Рекомендации

1. **Начинайте с простого** — не усложняйте без необходимости
2. **Доменная экспертиза важна** — создавайте осмысленные признаки
3. **Избегайте data leakage** — особенно при target encoding
4. **Валидируйте на кросс-валидации** — проверяйте устойчивость признаков
5. **Документируйте** — сохраняйте логику создания признаков
