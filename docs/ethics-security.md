# Этика и безопасность в ML/AI

## 1. Fairness (Справедливость)

### Типы смещений (Bias)

#### Historical Bias
```
Существующие социальные предрассудки в данных

Пример:
- Кредитные данные с дискриминацией по полу
- Найм с гендерным перекосом

Решение:
- Аудит данных
- Reweighting примеров
```

#### Representation Bias
```
Некоторые группы недостаточно представлены

Пример:
- Распознавание лиц, обученное на светлой коже
- Голосовые ассистенты, не понимающие акценты

Решение:
- Стратифицированная выборка
- Oversampling меньшинств
```

#### Measurement Bias
```
Некорректные или неполные признаки

Пример:
- Использование почтового индекса как proxy для расы
- Прокси-переменные для защищённых атрибутов

Решение:
- Анализ корреляций
- Удаление прокси-признаков
```

### Метрики Fairness

#### Demographic Parity
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)

Одинаковая доля положительных предсказаний
для всех групп (A - защищённый атрибут)
```

#### Equal Opportunity
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)

Одинаковый True Positive Rate для всех групп
```

#### Predictive Parity
```
P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1)

Одинаковая Precision для всех групп
```

### Техники обеспечения Fairness

#### Pre-processing
```
Reweighting:
- Изменение весов примеров
- Балансировка групп

Massaging:
- Изменение меток в обучающих данных
- Устранение исторических смещений
```

#### In-processing
```
Fairness constraints:
- Добавление ограничений в функцию потерь

Adversarial debiasing:
- Основная модель + adversarial для удаления bias
```

#### Post-processing
```
Threshold adjustment:
- Разные пороги для разных групп

Calibration:
- Калибровка вероятностей по группам
```

## 2. Interpretability (Интерпретируемость)

### Почему это важно?
```
- Доверие пользователей
- Отладка моделей
- Регуляторные требования (GDPR)
- Обнаружение bias
- Безопасность
```

### Global Interpretability

#### Feature Importance
```
Важность признаков для всей модели

Методы:
- Permutation importance
- Gini importance (деревья)
- SHAP values
```

#### Partial Dependence Plots (PDP)
```
Зависимость предсказания от признака
При усреднении по остальным признакам

Показывает:
- Направление влияния
- Линейность/нелинейность
```

#### SHAP (SHapley Additive exPlanations)
```
Основано на теории игр

φᵢ = Σ [f(S∪{i}) - f(S)] / C(n,|S|)

Преимущества:
- Локальная точность
- Consistency
- Global агрегация
```

### Local Interpretability

#### LIME (Local Interpretable Model-agnostic Explanations)
```
Аппроксимация сложной модели простой локально

Алгоритм:
1. Генерация пертурбированных примеров
2. Обучение интерпретируемой модели локально
3. Объяснение через веса простой модели
```

#### Counterfactual Explanations
```
"Что нужно изменить, чтобы получить другой результат?"

Пример:
- "Если бы доход был на 5000 больше, кредит одобрили бы"

Требования:
- Близость к исходному примеру
- Реалистичность
- Спarsity (минимум изменений)
```

## 3. Adversarial Attacks

### Типы атак

#### Evasion Attacks (Inference time)
```
Изменение входных данных для обмана модели

Примеры:
- Adversarial examples для изображений
- Незаметные изменения пикселей

Методы:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
```

#### Poisoning Attacks (Training time)
```
Внесение вредоносных данных в обучение

Цели:
- Снижение общей точности
- Backdoor (срабатывание на триггере)
- Targeted misclassification
```

#### Model Extraction
```
Кража модели через query

Атака:
1. Множественные запросы к API
2. Восстановление архитектуры и весов
3. Создание surrogate модели
```

#### Membership Inference
```
Определение, был ли пример в training set

Риск:
- Утечка приватной информации
- Нарушение GDPR
```

### Защита от атак

#### Adversarial Training
```
Обучение на adversarial примерах

min_θ max_δ L(f(x+δ), y)

Плюсы:
- Robustness к атакам

Минусы:
- Дорого вычислительно
- Может снизить accuracy на clean data
```

#### Defensive Distillation
```
Обучение мягкой модели на предсказаниях жесткой

Снижает чувствительность к малым изменениям
```

#### Input Validation
```
Детекция аномальных входов

Методы:
- Statistical tests
- Autoencoder reconstruction error
- Out-of-distribution detection
```

#### Differential Privacy
```
Гарантия, что выход не зависит от отдельных примеров

(ε, δ)-differential privacy:
P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ

Техники:
- Добавление шума к градиентам
- Gradient clipping
```

## 4. Privacy (Приватность)

### Риски приватности

#### Data Leakage
```
- Случайное включение таргета в признаки
- Информация из будущего в train
- Идентифицирующая информация
```

#### Model Inversion
```
Восстановление тренировочных данных из модели

Пример:
- Восстановление лиц из facial recognition модели
```

### Техники защиты

#### Federated Learning
```
Обучение без централизации данных

Принцип:
1. Модель отправляется на устройства
2. Обучение локально
3. Только обновления градиентов отправляются
4. Агрегация на сервере

Применение:
- Мобильные клавиатуры
- Медицинские данные
```

#### Homomorphic Encryption
```
Вычисления на зашифрованных данных

Типы:
- Partially homomorphic (одна операция)
- Somewhat homomorphic (ограниченное число)
- Fully homomorphic (любые операции)

Минусы:
- Вычислительная сложность
```

#### Secure Multi-Party Computation
```
Совместные вычисления без раскрытия данных

Принцип:
- Данные разделены между сторонами
- Вычисления без восстановления исходных данных
```

## 5. Regulatory Compliance

### GDPR (EU)
```
Key provisions:
- Right to explanation
- Right to erasure
- Data minimization
- Purpose limitation

Штрафы: до 4% глобального оборота
```

### CCPA (California)
```
- Right to know
- Right to delete
- Right to opt-out
- Non-discrimination
```

### AI Act (EU, proposed)
```
Risk-based approach:

Unacceptable risk: banned
- Social scoring
- Real-time remote biometric identification

High risk: strict requirements
- Critical infrastructure
- Employment decisions
- Law enforcement

Limited risk: transparency obligations
Minimal risk: no obligations
```

### Best Practices
```
1. Documentation
   - Model cards
   - Datasheets for datasets

2. Auditing
   - Regular fairness audits
   - Security assessments

3. Governance
   - Ethics review boards
   - Clear accountability

4. Transparency
   - Explain predictions to users
   - Disclose AI usage
```
