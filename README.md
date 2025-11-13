# Financial Sentiment Analysis - NLP

Текстова класифікація:
1) TF-IDF + Logistic Regression (MaxEnt) — scikit-learn
2) BiLSTM — TensorFlow/Keras
3) DistilBERT — transformers (готові ваги)


## Запуск
```bash
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python scripts/prepare_dataset.py

# 1) Baseline (скікит-льорн)
python -m src.baseline_sklearn --config config.json --limit 20000
# 2) BiLSTM (TensorFlow/Keras)
python -m src.lstm_keras --config config.json --limit 20000
# 3) DistilBERT (HF transformers)
python -m src.bert_transformer --config config.json --limit 20000
```

## TF-IDF + Logistic Regression (baseline sklearn)

- Що це за модель
Класична модель: перетворює текст у вектори TF-IDF, а потім лінійний класифікатор Logistic Regression (MaxEnt) прогнозує клас.

- Структура, компоненти, зв’язки
  - TfidfVectorizer - рахує вагу кожного слова в документі й корпусі.
  - LogisticRegression - бере TF-IDF-вектори та вчиться розділяти класи гіперплощиною.

- Як функціонує
  - На вході - сирий текст.
  - TF-IDF робить з тексту числовий вектор.
  - Logistic Regression рахує лінійну комбінацію ознак і дає ймовірність кожного класу (через sigmoid / softmax).
  - Обирається клас з найбільшою ймовірністю.

- Як навчається
  - TF-IDF будує словник по train-даних.
  - Logistic Regression мінімізує логістичну втрату (log loss) градієнтними методами.
  - Вага кожного слова коригується так, щоб правильно класифікувати train-приклади.

## BiLSTM (TensorFlow / Keras)

- Що це за модель
Послідовна нейромережа, яка читає текст у двох напрямках (Bidirectional LSTM) і класифікує його за сентиментом.

- Структура
  - TextVectorization - перетворює текст у послідовності індексів слів.
  - Embedding - вчить векторне представлення слів.
  - Bidirectional(LSTM) - читає послідовність зліва направо і справа наліво.

- Як функціонує
  - Текст - індекси токенів - вектори (Embedding).
  - BiLSTM проходиться по послідовності й формує вектор “сенсу” всієї фрази.
  - Dense + softmax перетворює цей вектор у ймовірності класів.

- Як навчається
  - Функція втрат: sparse_categorical_crossentropy.
  - Оптимізатор: Adam (градієнтний спуск).
  - Навчання батчами, з EarlyStopping по val_loss.
  - Ваги Embedding і LSTM оновлюються так, щоб зменшити втрату на train

## DistilBERT (Transformers, PyTorch)

- Що це за модель
Полегшений BERT-подібний трансформер, попередньо натренований на великому корпусі, який донавчається (fine-tuning) для задачі сентимент-аналізу.

- Структура
  - AutoTokenizer - перетворює текст у input_ids, attention_mask.
  - AutoModelForSequenceClassification (DistilBERT + класифікатор):
    - стек трансформер-шарів (self-attention + feed-forward),
    - зверху - лінійний класифікатор + softmax.

- Як функціонує
  - Tokenizer розбиває текст на субтокени, додає спеціальні токени, формує маски.
  - DistilBERT через self-attention дивиться на весь текст одразу й формує контекстний embedding.
  - Класифікатор на виході перетворює embedding у ймовірності класів.

- Як навчається
  - Fine-tuning: ми не тренуємо модель “з нуля”, а донавчаємо готові ваги.
  - Оптимізатор: torch.optim.AdamW.
  - Scheduler: get_linear_schedule_with_warmup (спочатку розігрів LR, потім лінійне зменшення).
  - Функція втрат: cross-entropy для багатокласової класифікації.
