# Перцептрон

Перцептрон — это базовая единица нейронной сети, математическая модель биологического нейрона.

## 1. Структура перцептрона

```
Входы: x = [x₁, x₂, ..., xₙ]
Веса: w = [w₁, w₂, ..., wₙ]
Смещение: b
Выход: y = f(w·x + b)
```

### Формула вычисления

```
z = Σ(wᵢ · xᵢ) + b = w·x + b
y = f(z)
```

где:
- `z` - взвешенная сумма входов (логит)
- `f` - функция активации
- `y` - выход перцептрона

## 2. Реализация на Python

### Базовая реализация
```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.01):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0
        self.lr = learning_rate
    
    def activate(self, z):
        """Функция активации (step function)"""
        return 1 if z >= 0 else 0
    
    def predict(self, X):
        """Прямое распространение"""
        z = np.dot(X, self.weights) + self.bias
        return self.activate(z)
    
    def fit(self, X, y, epochs=100):
        """Обучение перцептрона"""
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                
                # Обновление весов
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
```

### Векторизованная версия
```python
class PerceptronVectorized:
    def __init__(self, n_inputs, learning_rate=0.01):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0
        self.lr = learning_rate
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def fit(self, X, y, epochs=1000):
        m = len(y)
        for epoch in range(epochs):
            # Forward pass
            predictions = self.predict(X)
            
            # Вычисление градиентов
            error = predictions - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
            
            # Обновление параметров
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
```

## 3. Логические вентили

Перцептрон может реализовать линейно разделимые функции:

### AND (И)
```python
# Веса для AND
w = [1, 1], b = -1.5

# Таблица истинности
# x1 | x2 | y
# 0  | 0  | 0  → 1*0 + 1*0 - 1.5 = -1.5 < 0
# 0  | 1  | 0  → 1*0 + 1*1 - 1.5 = -0.5 < 0
# 1  | 0  | 0  → 1*1 + 1*0 - 1.5 = -0.5 < 0
# 1  | 1  | 1  → 1*1 + 1*1 - 1.5 = 0.5 >= 0
```

### OR (ИЛИ)
```python
# Веса для OR
w = [1, 1], b = -0.5

# Таблица истинности
# x1 | x2 | y
# 0  | 0  | 0  → 1*0 + 1*0 - 0.5 = -0.5 < 0
# 0  | 1  | 1  → 1*0 + 1*1 - 0.5 = 0.5 >= 0
# 1  | 0  | 1  → 1*1 + 1*0 - 0.5 = 0.5 >= 0
# 1  | 1  | 1  → 1*1 + 1*1 - 0.5 = 1.5 >= 0
```

### NOT (НЕ)
```python
# Веса для NOT(x1)
w = [-1], b = 0.5

# Таблица истинности
# x1 | y
# 0  | 1  → -1*0 + 0.5 = 0.5 >= 0
# 1  | 0  → -1*1 + 0.5 = -0.5 < 0
```

### XOR (Исключающее ИЛИ)
```
XOR НЕ реализуем одним перцептроном!
Требуется многослойная сеть.

# x1 | x2 | y
# 0  | 0  | 0
# 0  | 1  | 1
# 1  | 0  | 1
# 1  | 1  | 0  ← нелинейно разделимо
```

## 4. Геометрическая интерпретация

Перцептрон реализует **линейный классификатор**:

```
Решающая граница: w·x + b = 0

В 2D: w₁*x₁ + w₂*x₂ + b = 0  → прямая линия
В 3D: w₁*x₁ + w₂*x₂ + w₃*x₃ + b = 0  → плоскость
В n-D: гиперплоскость
```

### Визуализация
```python
import matplotlib.pyplot as plt

def plot_decision_boundary(perceptron, X, y):
    # Сетка для визуализации
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Предсказания на сетке
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Построение
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Decision Boundary')
    plt.show()
```

## 5. Ограничения перцептрона

### Теорема о сходимости
Если данные линейно разделимы, алгоритм перцептрона сойдется за конечное число шагов.

### Проблемы
1. **Только линейная разделимость** - не может решить XOR
2. **Нет вероятностной интерпретации** - выдает 0 или 1
3. **Чувствительность к выбросам** - одно неверное обновление
4. **Не сходится для неразделимых данных** - будет колебаться

### Решение
Использовать **многослойные сети** с нелинейными функциями активации.

## 6. Пример использования

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Генерация данных
X, y = make_classification(n_samples=100, n_features=2, 
                           n_redundant=0, n_informative=2,
                           random_state=42)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Обучение
perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron.fit(X_train, y_train, epochs=100)

# Оценка
predictions = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, (predictions >= 0.5).astype(int))
print(f"Accuracy: {accuracy:.2f}")
```

## Чек-лист понимания

- [ ] Понимаю формулу перцептрона: y = f(w·x + b)
- [ ] Могу реализовать перцептрон на Python
- [ ] Понимаю, какие функции реализует (AND, OR, NOT)
- [ ] Понимаю, почему XOR не реализуем
- [ ] Знаю геометрическую интерпретацию (гиперплоскость)
- [ ] Понимаю ограничения перцептрона

## Следующие шаги

Изучите [многослойные сети](multilayer.md) для решения нелинейных задач.
