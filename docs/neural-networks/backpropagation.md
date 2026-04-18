# Обратное распространение ошибки (Backpropagation)

## Введение

Обратное распространение ошибки (Backpropagation, Backprop) - это фундаментальный алгоритм обучения нейронных сетей. Он позволяет эффективно вычислять градиенты функции потерь по всем параметрам сети, используя цепное правило дифференцирования.

## Историческая справка

- **1974**: Пол Вербос описал алгоритм в своей диссертации
- **1986**: Дэвид Румельхарт, Джеффри Хинтон и Рональд Уильямс популяризировали алгоритм
- **Современность**: Основа всех современных методов глубокого обучения

## Интуиция

Представьте нейронную сеть как многоступенчатую систему:

1. **Прямой проход**: Данные проходят от входа к выходу, генерируя предсказание
2. **Вычисление ошибки**: Сравниваем предсказание с истинным значением
3. **Обратный проход**: Распространяем ошибку назад, определяя вклад каждого параметра
4. **Обновление весов**: Корректируем веса в направлении, уменьшающем ошибку

Ключевая идея: **цепное правило** позволяет выразить градиент сложной функции через градиенты её составляющих.

## Математические основы

### Цепное правило

Для композиции функций f(g(x)):
```
df/dx = df/dg · dg/dx
```

Для нейронной сети с множеством слоёв:
```
∂L/∂W[l] = ∂L/∂a[L] · ∂a[L]/∂z[L] · ∂z[L]/∂W[L] · ... · ∂a[l+1]/∂z[l+1] · ∂z[l]/∂W[l]
```

где:
- L - функция потерь
- W[l] - веса слоя l
- a[l] - активации слоя l
- z[l] - взвешенная сумма перед активацией

### Обозначения

- **l**: индекс слоя (1, 2, ..., L)
- **nₗ**: количество нейронов в слое l
- **W[l]**: матрица весов размера (nₗ, nₗ₋₁)
- **b[l]**: вектор смещений размера (nₗ, 1)
- **z[l]**: W[l]a[l-1] + b[l] - взвешенная сумма
- **a[l]**: f(z[l]) - активация после применения функции активации
- **a[0]**: x - входные данные
- **δ[l]**: ошибка слоя l

## Алгоритм обратного распространения

### Шаг 1: Прямой проход

Для каждого слоя l = 1, 2, ..., L:
```
z[l] = W[l] · a[l-1] + b[l]
a[l] = f(z[l])
```

Сохраняем все промежуточные значения для обратного прохода.

### Шаг 2: Вычисление ошибки на выходе

Для выходного слоя L:
```
δ[L] = ∂L/∂a[L] · f'(z[L])
```

Для разных функций потерь:

**MSE (среднеквадратичная ошибка):**
```
L = (1/n) Σ(a[L] - y)²
∂L/∂a[L] = (2/n)(a[L] - y)
δ[L] = (2/n)(a[L] - y) · f'(z[L])
```

**Cross-Entropy с softmax:**
```
δ[L] = a[L] - y  # упрощённая форма
```

### Шаг 3: Обратное распространение ошибки

Для слоёв l = L-1, L-2, ..., 1:
```
δ[l] = (W[l+1])ᵀ · δ[l+1] · f'(z[l])
```

Интерпретация:
- **(W[l+1])ᵀ · δ[l+1]**: распространение ошибки от следующего слоя
- **f'(z[l])**: учет локальной чувствительности функции активации

### Шаг 4: Вычисление градиентов

Градиенты по весам и смещениям:
```
∂L/∂W[l] = δ[l] · (a[l-1])ᵀ
∂L/∂b[l] = δ[l]
```

### Шаг 5: Обновление параметров

Градиентный спуск:
```
W[l] = W[l] - α · ∂L/∂W[l]
b[l] = b[l] - α · ∂L/∂b[l]
```

где α - learning rate.

## Полный алгоритм в псевдокоде

```python
def backpropagation(X, Y, parameters, caches, loss_type='mse'):
    L = len(parameters) // 2  # количество слоёв
    m = X.shape[1]  # количество примеров
    grads = {}
    
    # Шаг 1: Вычисление ошибки на выходе
    AL = caches[f'A{L}']
    
    if loss_type == 'mse':
        dAL = (2 / m) * (AL - Y)
    elif loss_type == 'cross_entropy':
        dAL = None  # обрабатывается отдельно
    
    ZL = caches[f'Z{L}']
    
    # Для cross-entropy + softmax
    if loss_type == 'cross_entropy':
        dZL = AL - Y  # комбинированный градиент
    else:
        dZL = dAL * sigmoid_derivative(ZL)  # или другая функция активации
    
    grads[f'dW{L}'] = (1/m) * np.dot(dZL, caches[f'A{L-1}'].T)
    grads[f'db{L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    
    # Шаг 2: Обратное распространение
    for l in reversed(range(1, L)):
        dA_prev = np.dot(parameters[f'W{l+1}'].T, dZL)
        
        Zl = caches[f'Z{l}']
        dZl = dA_prev * relu_derivative(Zl)  # или другая функция активации
        
        grads[f'dW{l}'] = (1/m) * np.dot(dZl, caches[f'A{l-1}'].T)
        grads[f'db{l}'] = (1/m) * np.sum(dZl, axis=1, keepdims=True)
        
        dZL = dZl
    
    return grads
```

## Пример: двухслойная сеть

Рассмотрим сеть с:
- Вход: x ∈ ℝⁿ
- Скрытый слой: h ∈ ℝᵐ с ReLU
- Выход: ŷ ∈ ℝ с sigmoid
- Функция потерь: Binary Cross-Entropy

### Прямой проход

```
z₁ = W₁x + b₁          # размер: (m, 1)
h = ReLU(z₁)           # размер: (m, 1)
z₂ = W₂h + b₂          # размер: (1, 1)
ŷ = sigmoid(z₂)        # размер: (1, 1)
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Обратный проход

**Выходной слой:**
```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
∂ŷ/∂z₂ = ŷ(1-ŷ)  # производная sigmoid

δ₂ = ∂L/∂z₂ = ∂L/∂ŷ · ∂ŷ/∂z₂
   = (-y/ŷ + (1-y)/(1-ŷ)) · ŷ(1-ŷ)
   = ŷ - y  # красивое упрощение!

∂L/∂W₂ = δ₂ · hᵀ = (ŷ - y) · hᵀ
∂L/∂b₂ = δ₂ = ŷ - y
```

**Скрытый слой:**
```
δ₂ = ŷ - y

∂L/∂h = W₂ᵀ · δ₂

∂h/∂z₁ = ReLU'(z₁) = 1 если z₁ > 0, иначе 0

δ₁ = (W₂ᵀ · δ₂) ⊙ ReLU'(z₁)  # ⊙ - поэлементное умножение

∂L/∂W₁ = δ₁ · xᵀ
∂L/∂b₁ = δ₁
```

## Реализация на Python

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = {}
        
        # Инициализация весов
        for l in range(1, len(layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]
            ) * np.sqrt(2.0 / layer_sizes[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    def forward(self, X):
        self.caches = {'A0': X}
        A = X
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            self.caches[f'Z{l}'] = Z
            self.caches[f'A{l}'] = A
        
        # Выходной слой
        ZL = np.dot(self.parameters[f'W{L}'], A) + self.parameters[f'b{L}']
        AL = self.sigmoid(ZL)
        self.caches[f'Z{L}'] = ZL
        self.caches[f'A{L}'] = AL
        
        return AL
    
    def compute_loss(self, AL, Y):
        m = Y.shape[1]
        loss = -np.mean(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15))
        return loss
    
    def backward(self, AL, Y):
        grads = {}
        L = len(self.parameters) // 2
        m = AL.shape[1]
        
        # Выходной слой (sigmoid + cross-entropy)
        dZL = AL - Y
        grads[f'dW{L}'] = (1/m) * np.dot(dZL, self.caches[f'A{L-1}'].T)
        grads[f'db{L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        
        # Скрытые слои
        dA_prev = np.dot(self.parameters[f'W{L}'].T, dZL)
        
        for l in reversed(range(1, L)):
            dZ = dA_prev * self.relu_derivative(self.caches[f'Z{l}'])
            grads[f'dW{l}'] = (1/m) * np.dot(dZ, self.caches[f'A{l-1}'].T)
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    def train(self, X, Y, epochs=1000, learning_rate=0.01, print_cost=True):
        costs = []
        
        for epoch in range(epochs):
            # Прямой проход
            AL = self.forward(X)
            
            # Вычисление потерь
            cost = self.compute_loss(AL, Y)
            
            # Обратный проход
            grads = self.backward(AL, Y)
            
            # Обновление параметров
            self.update_parameters(grads, learning_rate)
            
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
            costs.append(cost)
        
        return self.parameters, costs
```

## Векторизация

Одно из главных преимуществ backpropagation - возможность полной векторизации:

```python
# Вместо цикла по примерам
for i in range(m):
    # вычисления для одного примера

# Используем матричные операции
Z = np.dot(W, A_prev) + b  # сразу для всех примеров
```

Это даёт ускорение в 100-1000 раз на современном оборудовании.

## Проблемы и решения

### Исчезающие градиенты

**Проблема:** Градиенты становятся экспоненциально малыми при движении к нижним слоям.

**Причины:**
- Производные функций активации < 1 (sigmoid, tanh)
- Глубокие сети
- Плохая инициализация

**Решения:**
- ReLU и его вариации
- Правильная инициализация (Xavier, He)
- Batch Normalization
- Skip connections (ResNet)

### Взрывающиеся градиенты

**Проблема:** Градиенты становятся экспоненциально большими.

**Решения:**
- Gradient clipping: `grad = np.clip(grad, -threshold, threshold)`
- Правильная инициализация
- Batch Normalization

### Проверка градиентов (Gradient Checking)

Важно убедиться в правильности реализации backpropagation:

```python
def gradient_check(parameters, grads, X, Y, epsilon=1e-7):
    # Численное вычисление градиента
    numerical_grads = {}
    
    for key in parameters.keys():
        if not key.startswith('W'):
            continue
            
        param = parameters[key].flatten()
        grad_approx = np.zeros_like(param)
        
        for i in range(len(param)):
            # f(x + ε)
            param_plus = param.copy()
            param_plus[i] += epsilon
            parameters[key] = param_plus.reshape(parameters[key].shape)
            AL_plus = forward(X, parameters)
            loss_plus = compute_loss(AL_plus, Y)
            
            # f(x - ε)
            param_minus = param.copy()
            param_minus[i] -= epsilon
            parameters[key] = param_minus.reshape(parameters[key].shape)
            AL_minus = forward(X, parameters)
            loss_minus = compute_loss(AL_minus, Y)
            
            grad_approx[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        numerical_grads[key] = grad_approx.reshape(parameters[key].shape)
    
    # Восстановить оригинальные параметры
    # ...
    
    # Сравнение
    for key in parameters.keys():
        if not key.startswith('W'):
            continue
        
        diff = np.linalg.norm(numerical_grads[key] - grads['d' + key]) / \
               (np.linalg.norm(numerical_grads[key]) + np.linalg.norm(grads['d' + key]) + 1e-7)
        
        print(f"{key}: difference = {diff:.2e}")
        assert diff < 1e-7, f"Градиент для {key} не сходится!"
```

**Важно:** Gradient checking очень медленный, используйте только для отладки!

## Оптимизации

### Mini-batch Gradient Descent

Вместо использования всего датасета:
```python
# Разбиение на батчи
for batch_X, batch_Y in mini_batches:
    AL = forward(batch_X)
    grads = backward(AL, batch_Y)
    update_parameters(grads, learning_rate)
```

Преимущества:
- Быстрее сходимость
- Лучшая обобщающая способность
- Эффективное использование памяти

### Momentum

Добавление инерции к обновлениям:
```python
v_dW = β * v_dW + (1 - β) * dW
W = W - α * v_dW
```

где β ≈ 0.9

### Adam Optimizer

Комбинация Momentum и RMSProp:
```python
# Скользящее среднее градиентов
m_dW = β1 * m_dW + (1 - β1) * dW
# Скользящее среднее квадратов градиентов
v_dW = β2 * v_dW + (1 - β2) * (dW ** 2)

# Коррекция смещения
m_dW_corrected = m_dW / (1 - β1 ** t)
v_dW_corrected = v_dW / (1 - β2 ** t)

# Обновление
W = W - α * m_dW_corrected / (np.sqrt(v_dW_corrected) + ε)
```

## Визуализация процесса обучения

```python
import matplotlib.pyplot as plt

def plot_learning_curve(costs, learning_rate):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title(f'Функция потерь (learning rate = {learning_rate})')
    plt.xlabel('Эпоха (сотни)')
    plt.ylabel('Стоимость')
    plt.grid(True)
    plt.show()
```

## Заключение

Обратное распространение - это краеугольный камень глубокого обучения. Понимание этого алгоритма необходимо для:

- Отладки нейронных сетей
- Выбора правильных архитектур
- Оптимизации процесса обучения
- Разработки новых методов обучения

Хотя современные фреймворки (TensorFlow, PyTorch) автоматически вычисляют градиенты, глубокое понимание backpropagation помогает создавать более эффективные модели.

## Дополнительные ресурсы

- [Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/)
- [Stanford CS231n Notes](http://cs231n.github.io/optimization-2/)
- [3Blue1Brown - Backpropagation Visualization](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
