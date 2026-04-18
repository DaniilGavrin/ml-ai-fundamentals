# Трансформеры (Transformers)

## Введение

Трансформеры - это архитектура нейронных сетей, основанная на механизме внимания (attention mechanism), которая произвела революцию в обработке естественного языка и других областях.

## Почему Transformers?

### Ограничения RNN/LSTM

- Последовательная обработка (медленно)
- Проблемы с длинными зависимостями
- Сложность параллелизации

### Преимущества Transformers

- Полная параллелизация
- Прямые связи между любыми позициями
- Масштабируемость

## Архитектура Transformer

### Encoder-Decoder структура

```
Вход → [Encoder] → Контекст → [Decoder] → Выход
```

### Encoder

Состоит из N идентичных слоев:
- Multi-Head Attention
- Feed-Forward Network
- Layer Normalization
- Residual Connections

### Decoder

Также N слоев плюс:
- Masked Multi-Head Attention
- Encoder-Decoder Attention

## Self-Attention

### Механизм внимания

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

где:
- Q (Query) - запрос
- K (Key) - ключ
- V (Value) - значение
- d_k - размерность ключа

### Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    scores /= np.sqrt(d_k)
    weights = softmax(scores)
    output = np.matmul(weights, V)
    return output, weights
```

## Multi-Head Attention

Параллельное применение нескольких heads:

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

## Positional Encoding

Добавляет информацию о позиции:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)
```

## Feed-Forward Network

Позиционно-поэлементный полносвязный слой:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

## Полный Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.embedding_src = nn.Embedding(src_vocab, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab, d_model)
        self.pe = positional_encoding(max_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedded = self.dropout(self.embedding_src(src) + self.pe[:, :src.size(1)])
        tgt_embedded = self.dropout(self.embedding_tgt(tgt) + self.pe[:, :tgt.size(1)])
        
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)
        
        return self.fc_out(dec_output)
```

## Варианты архитектуры

### BERT (Bidirectional Encoder Representations)

- Только encoder
- Bidirectional attention
- Предобучение: MLM + NSP

### GPT (Generative Pre-trained Transformer)

- Только decoder
- Causal (masked) attention
- Авторегрессионная генерация

### T5 (Text-to-Text Transfer Transformer)

- Encoder-decoder
- Все задачи как text-to-text

### Vision Transformer (ViT)

- Применение к изображениям
- Patch embeddings вместо токенов

## Заключение

Transformers стали доминирующей архитектурой в NLP и активно применяются в компьютерном зрении, аудио и других областях.
