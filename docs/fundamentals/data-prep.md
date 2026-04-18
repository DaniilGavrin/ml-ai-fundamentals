# Подготовка данных

Подготовка данных — критически важный этап в машинном обучении, который часто занимает до 80% времени проекта.

## 1. Сбор данных

### Источники данных
- **Базы данных**: SQL, NoSQL хранилища
- **Файлы**: CSV, JSON, XML, Excel
- **API**: REST, GraphQL endpoints
- **Веб-скрапинг**: парсинг веб-страниц
- **Стриминг**: Kafka, RabbitMQ

### Пример загрузки данных
```python
import pandas as pd

# Из CSV
df = pd.read_csv('data.csv')

# Из Excel
df = pd.read_excel('data.xlsx')

# Из JSON
df = pd.read_json('data.json')

# Из SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM table', conn)
```

## 2. Очистка данных

### Обработка пропусков
```python
# Проверка пропусков
df.isnull().sum()

# Удаление строк с пропусками
df.dropna()

# Заполнение средним/медианой
df['column'].fillna(df['column'].mean(), inplace=True)
df['column'].fillna(df['column'].median(), inplace=True)

# Заполнение модой (для категориальных)
df['column'].fillna(df['column'].mode()[0], inplace=True)

# Интерполяция
df.interpolate(method='linear', inplace=True)
```

### Обработка дубликатов
```python
# Поиск дубликатов
df.duplicated().sum()

# Удаление дубликатов
df.drop_duplicates(inplace=True)
```

### Обработка выбросов
```python
from scipy import stats
import numpy as np

# Метод Z-score
z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]

# Метод IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['column'] < Q1 - 1.5*IQR) | (df['column'] > Q3 + 1.5*IQR)]

# Winsorization (ограничение выбросов)
from scipy.stats.mstats import winsorize
df['column'] = winsorize(df['column'], limits=[0.05, 0.05])
```

## 3. Преобразование данных

### Нормализация и стандартизация
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Мин-макс нормализация [0, 1]
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[['column']])

# Стандартизация (Z-score normalization)
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df[['column']])
```

### Кодирование категориальных признаков
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (для порядковых)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# One-Hot Encoding (для номинальных)
df_encoded = pd.get_dummies(df, columns=['category'])

# Target Encoding
def target_encode(df, column, target):
    mean_target = df.groupby(column)[target].mean()
    return df[column].map(mean_target)
```

### Преобразование признаков
```python
# Логарифмическое преобразование
df['log_column'] = np.log1p(df['column'])

# Квадратный корень
df['sqrt_column'] = np.sqrt(df['column'])

# Полиномиальные признаки
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

## 4. Разделение данных

```python
from sklearn.model_selection import train_test_split

# Базовое разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Для сохранения распределения классов
)

# Разделение на train/validation/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
```

## 5. Балансировка классов

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Комбинированный подход
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
```

## Чек-лист подготовки данных

- [ ] Проверить типы данных
- [ ] Обработать пропуски
- [ ] Удалить дубликаты
- [ ] Обработать выбросы
- [ ] Закодировать категориальные признаки
- [ ] Нормализовать/стандартизировать числовые признаки
- [ ] Сбалансировать классы (если нужно)
- [ ] Разделить на train/validation/test

## Рекомендации

1. **Всегда сохраняйте исходные данные** — работайте с копиями
2. **Документируйте все преобразования** — для воспроизводимости
3. **Проверяйте распределения** — до и после преобразований
4. **Избегайте data leakage** — fit на train, transform на test
5. **Автоматизируйте пайплайн** — используйте sklearn Pipeline
