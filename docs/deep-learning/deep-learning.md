# Глубокое обучение

## 1. Сверточные нейронные сети (CNN)

### Принцип работы
Используют свертки для извлечения локальных признаков из изображений.

**Операция свертки:**
```
Выход[i,j] = ΣΣ Вход[i+m, j+n] · Ядро[m, n] + bias
```

### Основные слои

#### Convolutional Layer
```
Параметры:
- Размер ядра (kernel size): 3×3, 5×5
- Количество фильтров
-Stride (шаг): 1, 2
- Padding: same, valid

Эффект:
- Извлечение признаков (края, текстуры, паттерны)
- Сохранение пространственной структуры
```

#### Pooling Layer
```
Max Pooling:
- Берёт максимум из области (обычно 2×2)
- Уменьшает размерность
- Повышает инвариантность к сдвигам

Average Pooling:
- Усреднение по области
- Более мягкое уменьшение
```

#### Fully Connected Layer
```
В конце сети для классификации
Объединяет все признаки
```

### Архитектуры CNN

#### LeNet-5 (1998)
```
Conv → Pool → Conv → Pool → FC → FC → Output
Для распознавания цифр
```

#### AlexNet (2012)
```
5 сверточных слоев + 3 полностью связных
ReLU, Dropout, Data Augmentation
Прорыв в ImageNet
```

#### VGG (2014)
```
Много слоев 3×3 вместо больших ядер
Глубина: 16-19 слоев
Простая и элегантная архитектура
```

#### ResNet (2015)
```
Остаточные блоки (Residual Blocks)
Skip connections: y = F(x) + x
Позволяет обучать очень глубокие сети (100+ слоев)
Решает проблему деградации
```

## 2. Рекуррентные нейронные сети (RNN)

### Базовая RNN
```
hₜ = f(W·hₜ₋₁ + U·xₜ + b)
yₜ = g(V·hₜ + c)

Проблемы:
- Затухающий градиент
- Взрывающийся градиент
- Не может запоминать длинные зависимости
```

### LSTM (Long Short-Term Memory)
```
Три гейта:
1. Forget Gate: fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)
2. Input Gate: iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)
3. Output Gate: oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)

Cell State: Cₜ = fₜ·Cₜ₋₁ + iₜ·tanh(W_C·[hₜ₋₁, xₜ] + b_C)
Hidden State: hₜ = oₜ·tanh(Cₜ)

Преимущества:
- Запоминание длинных зависимостей
- Контроль потока информации
```

### GRU (Gated Recurrent Unit)
```
Упрощенная версия LSTM
Два гейта:
1. Update Gate: zₜ = σ(W_z·[hₜ₋₁, xₜ])
2. Reset Gate: rₜ = σ(W_r·[hₜ₋₁, xₜ])

Новое состояние: h̃ₜ = tanh(W·[rₜ·hₜ₋₁, xₜ])
Выход: hₜ = (1-zₜ)·hₜ₋₁ + zₜ·h̃ₜ

Быстрее LSTM, сравнимое качество
```

### Bidirectional RNN
```
Обрабатывает последовательность в обоих направлениях
hₜ = [h→ₜ, h←ₜ]
Применение: NLP (контекст слева и справа)
```

## 3. Transformer и Attention Mechanisms

### Attention Mechanism
```
Query (Q), Key (K), Value (V)

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

√d_k - масштабирование для стабильных градиентов
```

### Self-Attention
```
Q, K, V получаются из одного входа
Каждое слово взаимодействует со всеми остальными
Параллельные вычисления (в отличие от RNN)
```

### Multi-Head Attention
```
Несколько attention "голов" параллельно
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)·W_O
headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)

Каждая голова учит разные типы зависимостей
```

### Transformer Encoder
```
1. Multi-Head Self-Attention
2. Add & Norm ( residual + LayerNorm)
3. Feed-Forward Network
4. Add & Norm

Повторяется N раз (обычно 6 или 12)
```

### Transformer Decoder
```
1. Masked Multi-Head Self-Attention
2. Multi-Head Cross-Attention (с encoder)
3. Feed-Forward Network
С residual connections и LayerNorm
```

### Positional Encoding
```
Добавляет информацию о позиции токенов
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Применения Transformer
```
- BERT (Encoder-only) - понимание текста
- GPT (Decoder-only) - генерация текста
- T5 (Encoder-Decoder) - перевод, суммаризация
- Vision Transformer (ViT) - классификация изображений
```

## 4. Generative Models

### Autoencoders
```
Encoder: z = f(x)  (сжатие)
Decoder: x̂ = g(z)  (восстановление)

Применения:
- Снижение размерности
- Denoising
- Generative модели
```

### Variational Autoencoders (VAE)
```
Кодирует в распределение N(μ, σ²)
Sample: z = μ + σ·ε, ε ~ N(0,1)
Loss = Reconstruction + KL-divergence

Генерирует новые данные
```

### GAN (Generative Adversarial Networks)
```
Generator: создаёт фейковые данные
Discriminator: отличает реальные от фейковых

Минимаксная игра:
min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]

Применения: генерация изображений, style transfer
```

### Diffusion Models
```
Forward process: постепенное добавление шума
Reverse process: обучение удалению шума

State-of-the-art для генерации изображений
DALL-E 2, Stable Diffusion, Midjourney
```
