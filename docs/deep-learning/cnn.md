# Сверточные нейронные сети (CNN)

## Введение

Сверточные нейронные сети (Convolutional Neural Networks, CNN, ConvNet) - это класс глубоких нейронных сетей, специально разработанных для обработки данных с сеточной структурой, таких как изображения.

## Почему CNN для изображений?

### Проблемы полносвязных сетей

Для изображения 256×256×3 (RGB):
- Количество входов: 256 × 256 × 3 = 196,608
- Для одного скрытого слоя с 1000 нейронов: 196,608 × 1000 ≈ 197 миллионов параметров!
- Переобучение, вычислительная сложность, потеря пространственной структуры

### Преимущества CNN

1. **Локальные связи** - нейроны соединены только с локальной областью входа
2. **Разделение весов** - одни и те же фильтры применяются ко всему изображению
3. **Пространственная иерархия** - от простых краев к сложным объектам
4. **Инвариантность к трансляции** - распознают объекты независимо от положения

## Архитектура CNN

### Основные слои

#### 1. Сверточный слой (Convolutional Layer)

Применяет набор фильтров (ядер) ко входному изображению:

```
Output = Input * Filter + bias
```

где * - операция свертки.

**Параметры:**
- **Количество фильтров (K)** - глубина выхода
- **Размер фильтра (F)** - обычно 3×3, 5×5, 7×7
- **Шаг (Stride, S)** - шаг скольжения фильтра
- **Дополнение (Padding, P)** - добавление границ

**Выходной размер:**
```
W_out = (W_in - F + 2P) / S + 1
H_out = (H_in - F + 2P) / S + 1
D_out = K
```

**Пример:**
- Вход: 32×32×3
- Фильтр: 5×5, stride=1, padding=0
- Количество фильтров: 6
- Выход: (32-5)/1+1 = 28 → 28×28×6

#### 2. Слой подвыборки (Pooling Layer)

Уменьшает пространственные размеры, сохраняя важную информацию:

**Max Pooling:**
```
Возвращает максимальное значение в области
```

**Average Pooling:**
```
Возвращает среднее значение в области
```

**Параметры:**
- Размер окна (обычно 2×2)
- Шаг (обычно 2)

**Выходной размер:**
```
W_out = (W_in - F) / S + 1
```

**Преимущества:**
- Уменьшение количества параметров
- Контроль переобучения
- Инвариантность к малым смещениям

#### 3. Полносвязный слой (Fully Connected Layer)

Обычно располагается в конце сети для классификации:
- Преобразует признаки в классы
- Аналогичен обычным нейронным сетям

#### 4. Нормализация (Batch Normalization)

Нормализует активации по батчу:
```
x_norm = (x - μ_batch) / √(σ²_batch + ε)
y = γ * x_norm + β
```

Ускоряет обучение, улучшает стабильность.

## Операция свертки

### 2D Свертка

Для входного изображения I и фильтра K:

```python
def convolve2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        yy, xx = image.shape
        y = yy - m + 1
        x = xx - n + 1
        z_new = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                z_new[i, j] = np.sum(image[i:i+m, j:j+n] * kernel)
        return z_new
```

### Пример фильтра

**Фильтр Собеля для детекции вертикальных краев:**
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Фильтр для детекции горизонтальных краев:**
```
[-1 -1 -1]
[ 0  0  0]
[ 1  1  1]
```

### Многоканальная свертка

Для RGB изображения (3 канала):
- Фильтр имеет размер F×F×3
- Свертка применяется по всем каналам
- Результаты суммируются

```
Output[i,j] = Σₖ Σₘ Σₙ Input[i+m, j+n, k] × Filter[m, n, k]
```

## Типы сверток

### 1. Обычная свертка (Valid Padding)

Без дополнения границ:
```
W_out = W_in - F + 1
```

### 2. Свертка с дополнением (Same Padding)

Добавление нулей по краям для сохранения размера:
```
P = (F - 1) / 2  # для нечетного F
W_out = W_in
```

### 3. Транспонированная свертка (Transposed Convolution)

Увеличивает пространственные размеры:
- Используется в сегментации изображений
- Генеративных моделях

### 4. Разделенная свертка (Depthwise Separable Convolution)

Разделяет свертку на два этапа:
1. **Depthwise**: свертка по каждому каналу отдельно
2. **Pointwise**: свертка 1×1 для комбинации каналов

Экономит параметры, используется в MobileNet.

### 5. Дилатированная свертка (Dilated Convolution)

Фильтр с пропусками (atrous convolution):
```
Увеличивает receptive field без увеличения параметров
```

Используется в семантической сегментации.

## Receptive Field

Receptive field - область входного изображения, влияющая на конкретный нейрон.

**Расчет:**
```
RF_l = RF_{l-1} + (F_l - 1) × Π_{i=1}^{l-1} S_i
```

где:
- F_l - размер фильтра на слое l
- S_i - stride на предыдущих слоях

**Пример:**
- Conv1: 3×3, stride=1 → RF = 3
- Conv2: 3×3, stride=1 → RF = 5
- Pool: 2×2, stride=2 → RF = 12

Чем больше receptive field, тем более глобальный контекст учитывает нейрон.

## Классические архитектуры CNN

### LeNet-5 (1998)

Пионер CNN для распознавания цифр:
```
Input → Conv → Pool → Conv → Pool → FC → FC → Output
```

### AlexNet (2012)

Прорыв в ImageNet:
- 8 слоёв (5 conv + 3 FC)
- ReLU активации
- Dropout
- Data augmentation

### VGGNet (2014)

Глубокая сеть с однородной архитектурой:
- Только фильтры 3×3
- Глубина 16-19 слоёв
- Простая и элегантная архитектура

### GoogLeNet / Inception (2014)

Модули Inception с параллельными свертками:
```
[1×1 conv]
[3×3 conv]     → concatenate
[5×5 conv]
[max pool + 1×1]
```

### ResNet (2015)

Остаточные соединения (skip connections):
```
Output = F(x) + x
```

Позволяет обучать очень глубокие сети (до 152 слоёв).

### EfficientNet (2019)

Масштабирование по глубине, ширине и разрешению:
- Compound scaling
- State-of-the-art эффективность

## Реализация CNN на Python

```python
import numpy as np

class ConvLayer:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # Инициализация весов (Xavier)
        scale = np.sqrt(2.0 / (in_channels * filter_size * filter_size))
        self.weights = np.random.randn(
            out_channels, in_channels, filter_size, filter_size
        ) * scale
        self.biases = np.zeros(out_channels)
        
        self.cache = None
    
    def forward(self, X):
        batch_size = X.shape[0]
        
        # Добавление padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
        
        # Вычисление выходных размеров
        _, _, h_in, w_in = X_padded.shape
        h_out = (h_in - self.filter_size) // self.stride + 1
        w_out = (w_in - self.filter_size) // self.stride + 1
        
        # Инициализация выхода
        out = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Свертка
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                # Извлечение региона
                region = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Свертка для каждого фильтра
                for f in range(self.out_channels):
                    out[:, f, i, j] = np.sum(region * self.weights[f], axis=(1, 2, 3))
                
                out[:, :, i, j] += self.biases
        
        self.cache = (X, X_padded, region)
        return out
    
    def backward(self, dout):
        X, X_padded, _ = self.cache
        batch_size = X.shape[0]
        
        # Градиенты
        dweights = np.zeros_like(self.weights)
        dbiases = np.sum(dout, axis=(0, 2, 3))
        dX_padded = np.zeros_like(X_padded)
        
        # Вычисление градиентов
        _, _, h_out, w_out = dout.shape
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                region = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                for f in range(self.out_channels):
                    dweights[f] += np.sum(
                        region * dout[:, f, i, j][:, None, None, None],
                        axis=0
                    )
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[f] * dout[:, f, i, j][:, None, None, None]
        
        # Удаление padding из градиента
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX, dweights, dbiases


class MaxPoolLayer:
    def __init__(self, filter_size=2, stride=2):
        self.filter_size = filter_size
        self.stride = stride
        self.cache = None
    
    def forward(self, X):
        batch_size, channels, h_in, w_in = X.shape
        
        h_out = (h_in - self.filter_size) // self.stride + 1
        w_out = (w_in - self.filter_size) // self.stride + 1
        
        out = np.zeros((batch_size, channels, h_out, w_out))
        max_indices = {}
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        region = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        
                        out[b, c, i, j] = max_val
                        max_indices[(b, c, i, j)] = (h_start + max_idx[0], 
                                                     w_start + max_idx[1])
        
        self.cache = (X, max_indices)
        return out
    
    def backward(self, dout):
        X, max_indices = self.cache
        batch_size, channels, h_in, w_in = X.shape
        
        dX = np.zeros_like(X)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(dout.shape[2]):
                    for j in range(dout.shape[3]):
                        h, w = max_indices[(b, c, i, j)]
                        dX[b, c, h, w] += dout[b, c, i, j]
        
        return dX
```

## Data Augmentation

Методы увеличения данных для улучшения обобщения:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,        # случайные повороты
    width_shift_range=0.2,    # сдвиги по ширине
    height_shift_range=0.2,   # сдвиги по высоте
    shear_range=0.2,          # искажения
    zoom_range=0.2,           # зум
    horizontal_flip=True,     # горизонтальные отражения
    fill_mode='nearest'       # заполнение новых пикселей
)
```

## Визуализация признаков

```python
import matplotlib.pyplot as plt

def visualize_filters(conv_layer):
    """Визуализация фильтров сверточного слоя"""
    filters = conv_layer.weights
    n_filters = filters.shape[0]
    
    fig, axes = plt.subplots(n_filters // 8, 8, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Нормализация для визуализации
            f = (filters[i] - filters[i].min()) / (filters[i].max() - filters[i].min())
            ax.imshow(f.transpose(1, 2, 0))
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(conv_output):
    """Визуализация карт признаков"""
    n_maps = conv_output.shape[1]
    
    fig, axes = plt.subplots(n_maps // 8, 8, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(conv_output[0, i], cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Практические рекомендации

### Выбор архитектуры

1. **Начните с простой модели**
   - Несколько сверточных слоев
   - Постепенно увеличивайте глубину

2. **Используйте предобученные модели**
   - Transfer learning
   - Fine-tuning под вашу задачу

3. **Размер фильтра**
   - 3×3 - стандартный выбор
   - 1×1 - для изменения глубины
   - 5×5, 7×7 - только в первых слоях

### Гиперпараметры

- **Learning rate**: 0.001 - 0.01 (с decay)
- **Batch size**: 32, 64, 128
- **Optimizer**: Adam, SGD with momentum
- **Regularization**: Dropout, L2, data augmentation

### Предобработка данных

1. Нормализация пикселей (0-1 или -1 до 1)
2. Вычитание среднего значения
3. Resize к единому размеру

## Заключение

CNN произвели революцию в компьютерном зрении и продолжают доминировать во многих задачах:
- Классификация изображений
- Детекция объектов
- Семантическая сегментация
- Распознавание лиц
- Медицинская диагностика

Понимание принципов работы CNN необходимо для любого специалиста по машинному обучению.

## Дополнительные ресурсы

- [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book - Chapter 9](https://www.deeplearningbook.org/)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Keras CNN Guide](https://keras.io/guides/sequential_model/)
