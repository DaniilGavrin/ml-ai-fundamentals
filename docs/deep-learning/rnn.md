# Рекуррентные нейронные сети (RNN)

## Введение

Рекуррентные нейронные сети (Recurrent Neural Networks, RNN) - это класс нейронных сетей, предназначенных для обработки последовательных данных, где порядок элементов имеет значение.

## Почему RNN?

### Ограничения обычных сетей

Обычные нейронные сети (CNN, FC) предполагают:
- Независимость входов друг от друга
- Фиксированный размер входа
- Отсутствие памяти о предыдущих входах

### Задачи с последовательностями

- **Временные ряды**: прогнозирование, анализ
- **Текст**: перевод, генерация, классификация
- **Речь**: распознавание, синтез
- **Видео**: анализ действий, предсказание

## Архитектура RNN

### Базовая ячейка RNN

```
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = g(W_hy · h_t + b_y)
```

где:
- x_t - вход в момент времени t
- h_t - скрытое состояние в момент t
- y_t - выход в момент t
- W_hh, W_xh, W_hy - матрицы весов
- f, g - функции активации

### Развертка во времени

RNN можно представить как развернутую сеть:

```
x_0 → [RNN] → h_0 → y_0
        ↓
x_1 → [RNN] → h_1 → y_1
        ↓
x_2 → [RNN] → h_2 → y_2
```

Одни и те же веса применяются на каждом шаге.

## Типы архитектур RNN

### One-to-One

Обычная нейронная сеть:
```
Вход → [RNN] → Выход
```
Пример: классификация изображений

### One-to-Many

Один вход, последовательность выходов:
```
Вход → [RNN] → Выход_1 → Выход_2 → ...
```
Пример: генерация подписей к изображениям

### Many-to-One

Последовательность входов, один выход:
```
Вход_1 → Вход_2 → ... → [RNN] → Выход
```
Пример: классификация текста, анализ тональности

### Many-to-Many (синхронный)

```
Вход_1 → [RNN] → Выход_1
Вход_2 → [RNN] → Выход_2
```
Пример: POS-tagging

### Many-to-Many (асинхронный)

```
Вход_1 → Вход_2 → [RNN] → Выход_1 → Выход_2
```
Пример: машинный перевод

## Проблемы базовых RNN

### Исчезающие градиенты

При обратном распространении через много шагов:
- Градиенты умножаются на каждом шаге
- При |W| < 1 градиенты экспоненциально затухают
- Сеть не может обучаться долгосрочным зависимостям

### Взрывающиеся градиенты

При |W| > 1 градиенты экспоненциально растут.

**Решение**: Gradient clipping
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

## LSTM (Long Short-Term Memory)

LSTM решает проблему исчезающих градиентов с помощью гейтов.

### Архитектура LSTM

```python
# Forget gate: что забыть
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Input gate: что запомнить
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

# Update cell state
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

# Output gate: что выдать
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

где:
- c_t - состояние ячейки (cell state)
- h_t - скрытое состояние
- σ - sigmoid функция
- ⊙ - поэлементное умножение

### Преимущества LSTM

- Явная память через cell state
- Гейты контролируют поток информации
- Могут запоминать на сотни шагов

## GRU (Gated Recurrent Unit)

Упрощенная версия LSTM:

```python
# Reset gate: насколько забыть прошлое
r_t = σ(W_r · [h_{t-1}, x_t])

# Update gate: насколько обновить
z_t = σ(W_z · [h_{t-1}, x_t])

# Candidate activation
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])

# Final hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### Сравнение LSTM и GRU

| Характеристика | LSTM | GRU |
|---------------|------|-----|
| Параметров | Больше | Меньше |
| Скорость | Медленнее | Быстрее |
| Качество | Немного лучше | Почти такое же |
| Рекомендация | Большие данные | Ограниченные ресурсы |

## Bidirectional RNN

Обрабатывает последовательность в обоих направлениях:

```
Вперед: x_1 → x_2 → ... → x_n
Назад:  x_n → x_{n-1} → ... → x_1
Выход:  Concat(h_forward, h_backward)
```

Применение: NLP задачи, где контекст важен с обеих сторон.

## Deep RNN

Многослойные RNN для более сложных представлений:

```
x → [RNN L1] → h_1 → [RNN L2] → h_2 → ... → Выход
```

Каждый слой изучает разные уровни абстракции.

## Реализация RNN на Python

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Инициализация весов
        scale = 0.1
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_hy = np.random.randn(hidden_size, output_size) * scale
        
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        self.cache = {}
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """
        X: последовательность входов (batch_size, seq_length, input_size)
        """
        batch_size, seq_length, _ = X.shape
        
        # Инициализация
        h = np.zeros((batch_size, self.hidden_size))
        self.cache['X'] = X
        self.cache['h'] = [h]
        
        # Прямой проход по времени
        for t in range(seq_length):
            x_t = X[:, t, :]
            h = self.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            self.cache['h'].append(h)
        
        # Выход
        y = self.softmax(np.dot(h, self.W_hy) + self.b_y)
        
        return y, h
    
    def backward(self, dy, learning_rate=0.01):
        batch_size = dy.shape[0]
        X = self.cache['X']
        seq_length = X.shape[1]
        hs = self.cache['h']
        
        # Инициализация градиентов
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Градиент по выходному слою
        dh_next = np.dot(dy, self.W_hy.T)
        dW_hy = np.dot(hs[-1].T, dy)
        db_y = np.sum(dy, axis=0, keepdims=True)
        
        # Обратное распространение во времени (BPTT)
        for t in reversed(range(seq_length)):
            dh = dh_next * self.tanh_derivative(np.dot(X[:, t], self.W_xh) + 
                                                 np.dot(hs[t], self.W_hh) + self.b_h)
            
            dW_xh += np.dot(X[:, t].T, dh)
            dW_hh += np.dot(hs[t].T, dh)
            db_h += np.sum(dh, axis=0, keepdims=True)
            
            dh_next = np.dot(dh, self.W_hh.T)
        
        # Обновление весов (градиентный спуск)
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Веса для каждого гейта
        scale = 0.1
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * scale
        
        self.b_f = np.ones((1, hidden_size))  # forget gate инициализируется единицами
        self.b_i = np.zeros((1, hidden_size))
        self.b_c = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))
        
        self.cache = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X, h_prev=None, c_prev=None):
        batch_size, seq_length, _ = X.shape
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
        if c_prev is None:
            c_prev = np.zeros((batch_size, self.hidden_size))
        
        self.cache['X'] = X
        self.cache['h'] = [h_prev]
        self.cache['c'] = [c_prev]
        
        h = h_prev
        c = c_prev
        
        for t in range(seq_length):
            x_t = X[:, t, :]
            combined = np.concatenate([h, x_t], axis=1)
            
            # Gates
            f_t = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
            i_t = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
            c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)
            o_t = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
            
            # Update states
            c = f_t * c + i_t * c_tilde
            h = o_t * np.tanh(c)
            
            self.cache['h'].append(h)
            self.cache['c'].append(c)
        
        return h, c
```

## Attention Mechanism

Механизм внимания позволяет модели фокусироваться на релевантных частях входа:

```python
def attention(query, key, value):
    scores = np.matmul(query, key.transpose(0, 2, 1))
    scores /= np.sqrt(key.shape[-1])
    weights = softmax(scores)
    output = np.matmul(weights, value)
    return output, weights
```

## Transformer Architecture

Современная архитектура на основе attention:

- Self-Attention для контекста
- Multi-Head Attention для разных представлений
- Positional Encoding для порядка
- Feed-Forward слои
- Layer Normalization и residual connections

## Практическое применение

```python
# PyTorch пример
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Используем последнее скрытое состояние с обоих направлений
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))
```

## Заключение

RNN и их варианты (LSTM, GRU) остаются важными инструментами для работы с последовательностями, хотя в некоторых задачах их вытесняют Transformers.

## Дополнительные ресурсы

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
