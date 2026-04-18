# Автоэнкодеры (Autoencoders)

## Введение

Автоэнкодеры - это тип нейронных сетей для обучения без учителя, которые учатся сжимать и восстанавливать данные.

## Архитектура

```
Вход → [Encoder] → Latent Space → [Decoder] → Выход
```

### Encoder

Сжимает вход в латентное представление:
```
h = f(W_e · x + b_e)
```

### Decoder

Восстанавливает из латентного представления:
```
x̂ = g(W_d · h + b_d)
```

### Функция потерь

Минимизируется разница между входом и выходом:
```
L = ||x - x̂||²
```

## Типы автоэнкодеров

### Undercomplete Autoencoder

Латентное пространство меньше входа:
```
dim(z) < dim(x)
```

Принуждает сеть изучать важные признаки.

### Sparse Autoencoder

Добавляется штраф за активацию:
```
L = ||x - x̂||² + λ · Σ|h|
```

### Denoising Autoencoder

Обучается восстанавливать чистые данные из зашумленных:
```
x̃ = x + noise
x̂ = decoder(encoder(x̃))
L = ||x - x̂||²
```

### Variational Autoencoder (VAE)

Кодирует распределение в латентном пространстве:

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def loss_function(self, x, x_hat, mu, logvar):
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence
```

## Применения

- Снижение размерности
- Удаление шума
- Генерация данных
- Аномалии детекция
- Рекомендательные системы

## Заключение

Автоэнкодеры - мощный инструмент для обучения представлений без размеченных данных.
