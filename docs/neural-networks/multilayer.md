# Многослойные нейронные сети

## Введение

Многослойные нейронные сети (Multilayer Neural Networks) представляют собой архитектуру, состоящую из нескольких последовательно соединённых слоёв нейронов. Такая структура позволяет сети обучаться сложным нелинейным зависимостям в данных.

## Архитектура многослойной сети

### Слои сети

1. **Входной слой (Input Layer)**
   - Принимает исходные данные
   - Количество нейронов равно размерности входных данных
   - Не выполняет вычислений, только передаёт данные

2. **Скрытые слои (Hidden Layers)**
   - Один или несколько слоёв между входом и выходом
   - Каждый нейрон связан со всеми нейронами предыдущего слоя
   - Выполняют основные вычисления и извлечение признаков

3. **Выходной слой (Output Layer)**
   - Produces the final output
   - Количество нейронов зависит от задачи (1 для регрессии, N для классификации)

### Полносвязная архитектура

В полносвязной сети (Fully Connected / Dense) каждый нейрон слоя l связан со всеми нейронами слоя l-1:

```
y = f(Wx + b)
```

где:
- x - входной вектор
- W - матрица весов
- b - вектор смещений
- f - функция активации

## Глубокие нейронные сети

Глубокая нейронная сеть (Deep Neural Network, DNN) содержит множество скрытых слоёв:

- **Shallow networks**: 1-2 скрытых слоя
- **Deep networks**: 3+ скрытых слоя
- **Very deep networks**: 10+ слоёв (например, ResNet-152)

### Преимущества глубины

1. **Иерархическое представление**
   - Нижние слои учатся простым признакам
   - Верхние слои комбинируют их в сложные концепции

2. **Эффективность параметров**
   - Глубокие сети требуют меньше параметров для той же выразительности
   - Экспоненциальное увеличение представимости с глубиной

3. **Автоматическое извлечение признаков**
   - Не требуется ручная инженерия признаков
   - Сеть сама учится релевантным представлениям

## Прямое распространение (Forward Propagation)

Процесс вычисления выхода сети:

```python
def forward_propagation(X, parameters):
    L = len(parameters) // 2  # количество слоёв
    
    A = X
    caches = []
    
    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = activation_function(Z)  # например, ReLU
        caches.append((A_prev, Z))
    
    # Выходной слой
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = output_activation(ZL)  # например, softmax
    
    return AL, caches
```

## Инициализация весов

Правильная инициализация критична для обучения глубоких сетей:

### Методы инициализации

1. **Xavier/Glorot Initialization**
   ```python
   W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
   ```
   - Хорошо работает с tanh и sigmoid

2. **He Initialization**
   ```python
   W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
   ```
   - Рекомендуется для ReLU и её вариаций

3. **LeCun Initialization**
   ```python
   W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
   ```
   - Для SELU активации

## Проблемы глубоких сетей

### Исчезающие градиенты (Vanishing Gradients)

При обратном распространении градиенты могут становиться экспоненциально малыми:

- Градиент умножается на каждом слое
- При малых производных функций активации градиент затухает
- Нижние слои перестают обучаться

**Решения:**
- ReLU и её вариации вместо sigmoid/tanh
- Правильная инициализация весов
- Batch Normalization
- Skip connections (Residual Networks)

### Взрывающиеся градиенты (Exploding Gradients)

Противоположная проблема - градиенты становятся слишком большими:

- Веса обновляются на огромные величины
- Обучение становится нестабильным

**Решения:**
- Gradient clipping
- Правильная инициализация
- Batch Normalization

## Регуляризация в многослойных сетях

### Dropout

Случайное "отключение" нейронов во время обучения:

```python
def apply_dropout(A, keep_prob):
    D = np.random.rand(*A.shape) < keep_prob
    A = A * D
    A = A / keep_prob  # масштабирование
    return A
```

### L1/L2 Регуляризация

Добавление штрафа к функции потерь:

- **L1**: `loss += λ * Σ|w|` (способствует разреженности)
- **L2**: `loss += λ * Σw²` (предотвращает большие веса)

### Batch Normalization

Нормализация активаций каждого слоя:

```
μ = mean(x)
σ² = variance(x)
x_norm = (x - μ) / √(σ² + ε)
y = γ * x_norm + β
```

Преимущества:
- Ускоряет обучение
- Снижает чувствительность к инициализации
- Действует как регуляризатор

## Практические рекомендации

### Выбор архитектуры

1. **Начните просто**
   - Начните с 1-2 скрытых слоёв
   - Постепенно увеличивайте глубину при необходимости

2. **Размер слоёв**
   - Обычно уменьшается к выходу (пирамидальная структура)
   - Или остаётся постоянным

3. **Количество нейронов**
   - Между размером входа и выхода
   - Эмпирически: 2/3 размера входа + размер выхода

### Гиперпараметры

- **Learning rate**: 0.001 - 0.1 (начните с 0.01)
- **Batch size**: 32, 64, 128, 256
- **Количество эпох**: используйте early stopping
- **Optimizer**: Adam обычно лучший выбор

### Мониторинг обучения

Отслеживайте:
- Loss на training и validation наборах
- Точность/метрики качества
- Градиенты и веса (для отладки)

## Пример реализации на Python

```python
import numpy as np

class MultiLayerNN:
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.activation = activation
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
    
    def forward(self, X):
        self.caches = {'A0': X}
        A = X
        
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            self.caches[f'Z{l}'] = Z
            self.caches[f'A{l}'] = A
        
        # Выходной слой (sigmoid для бинарной классификации)
        ZL = np.dot(self.parameters[f'W{L}'], A) + self.parameters[f'b{L}']
        AL = 1 / (1 + np.exp(-ZL))
        self.caches[f'Z{L}'] = ZL
        self.caches[f'A{L}'] = AL
        
        return AL
    
    def backward(self, AL, Y):
        grads = {}
        L = len(self.parameters) // 2
        m = AL.shape[1]
        
        # Выходной слой
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
    
    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            AL = self.forward(X)
            grads = self.backward(AL, Y)
            self.update_parameters(grads, learning_rate)
            
            if epoch % 100 == 0:
                loss = -np.mean(Y * np.log(AL) + (1-Y) * np.log(1-AL))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Заключение

Многослойные нейронные сети являются фундаментом глубокого обучения. Понимание их архитектуры, процесса обучения и методов регуляризации необходимо для построения эффективных моделей.

## Дополнительные ресурсы

- [Deep Learning Book](https://www.deeplearningbook.org/) - глава 6
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Stanford CS231n](http://cs231n.stanford.edu/) - лекции о нейронных сетях
