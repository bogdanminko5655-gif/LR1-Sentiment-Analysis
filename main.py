"""ЛР №1. Логістична регресія для аналізу тональності (IMDb / UCI).

Запуск: python main.py
Власний корпус TSV: python main.py --data data/my_data.txt --no-download
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data' / 'imdb_labelled.txt'
RESULTS_DIR = ROOT / 'results'
DATASET_URL = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
STEMMER = PorterStemmer()
# Не видаляємо заперечення: вони мають значення для тональності.
STOPWORDS = set(ENGLISH_STOP_WORDS) - {'no', 'nor', 'not', 'never'}


def download_imdb_dataset(destination: Path) -> None:
    """Завантажує саме офіційний архів UCI, не розпаковуючи сторонні шляхи."""
    from io import BytesIO
    print('Завантаження IMDb з UCI Machine Learning Repository...')
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=40) as response:
            content = response.read()
        with zipfile.ZipFile(BytesIO(content)) as archive:
            target = next(
                (name for name in archive.namelist() if name.replace('\\', '/').endswith('/imdb_labelled.txt')),
                None,
            )
            if target is None:
                raise ValueError('У завантаженому архіві немає imdb_labelled.txt')
            dataset = archive.read(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(dataset)
    except Exception as exc:
        raise RuntimeError(
            'Не вдалося завантажити дані. Завантажте архів вручну з '
            'https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences '
            'і покладіть imdb_labelled.txt у папку data/. '
            'Або передайте шлях через --data.'
        ) from exc


def load_corpus(filename: Path) -> tuple[list[str], np.ndarray]:
    sentences, labels = [], []
    with filename.open('r', encoding='utf-8-sig') as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sentence, value = line.rstrip('\n\r').rsplit('\t', 1)
                sentiment = int(value.strip())
                if not sentence.strip() or sentiment not in (0, 1):
                    raise ValueError('Порожній текст або неправильний клас')
            except Exception as exc:
                raise ValueError(f'Помилка у рядку {row_number}: очікується текст<TAB>0/1') from exc
            sentences.append(sentence.strip())
            labels.append(sentiment)
    if len(sentences) < 30 or set(labels) != {0, 1}:
        raise ValueError('Для експерименту потрібно >=30 прикладів і обидва класи (0, 1).')
    return sentences, np.asarray(labels, dtype=int)


def process_text(text: str) -> list[str]:
    """Нормалізація -> токенізація -> стоп-слова -> стемінг."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r"n['’]t\b", ' not', text)  # didn't -> did not
    text = re.sub(r"\bwon['’]t\b", 'will not', text)
    text = re.sub(r"\bcan['’]t\b", 'can not', text)
    # Підтримується англомовний корпус; для українського потрібна інша морфологія.
    tokens = re.findall(r'[a-z]+', text)
    return [STEMMER.stem(word) for word in tokens if word not in STOPWORDS and len(word) > 1]


def build_freqs(texts: list[str], labels: np.ndarray) -> Counter:
    frequencies = Counter()
    for text, label in zip(texts, labels):
        for word in process_text(text):
            frequencies[(word, int(label))] += 1
    return frequencies


def extract_features(text: str, freqs: Counter) -> np.ndarray:
    """[1, сума частот позитивного класу, сума частот негативного класу]."""
    vector = np.ones(3, dtype=float)
    vector[1:] = 0.0
    for word in process_text(text):
        vector[1] += freqs.get((word, 1), 0)
        vector[2] += freqs.get((word, 0), 0)
    return vector


def feature_matrix(texts: list[str], freqs: Counter) -> np.ndarray:
    return np.vstack([extract_features(text, freqs) for text in texts])


class FeatureScaler:
    """Згладжує великі частоти, масштабує лише 2 числові ознаки."""
    def __init__(self):
        self.mean = np.zeros(2)
        self.std = np.ones(2)

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        log_features = np.log1p(X[:, 1:])
        self.mean = log_features.mean(axis=0)
        self.std = np.maximum(log_features.std(axis=0), 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed = X.copy().astype(float)
        transformed[:, 1:] = (np.log1p(X[:, 1:]) - self.mean) / self.std
        return transformed


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.clip(np.asarray(p).reshape(-1), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0-y) * np.log(1.0-p)))


def gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, num_iters: int,
                     save_every: int = 1) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Власна реалізація повнопакетного градієнтного спуску."""
    if alpha <= 0 or num_iters < 1:
        raise ValueError('Швидкість навчання та кількість ітерацій мають бути > 0')
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.zeros(X.shape[1], dtype=float)
    history = []
    for iteration in range(1, num_iters + 1):
        probabilities = sigmoid(X @ theta)
        gradient = (X.T @ (probabilities - y)) / len(y)
        theta -= alpha * gradient
        if iteration == 1 or iteration % save_every == 0 or iteration == num_iters:
            history.append((iteration, log_loss(y, sigmoid(X @ theta))))
    return theta, history


def score(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> dict:
    p = sigmoid(X @ theta)
    pred = (p >= 0.5).astype(int)
    y = np.asarray(y).reshape(-1)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return dict(accuracy=float(np.mean(pred == y)), loss=log_loss(y, p),
                precision=precision, recall=recall, f1=f1,
                confusion_matrix=[[tn, fp], [fn, tp]])


def prepare(texts: list[str], labels: np.ndarray) -> tuple[Counter, FeatureScaler, np.ndarray]:
    freqs = build_freqs(texts, labels)
    X = feature_matrix(texts, freqs)
    scaler = FeatureScaler().fit(X)
    return freqs, scaler, scaler.transform(X)


def save_csv(path: Path, headers: list[str], rows: list) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_history(history: list[tuple[int, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot([item[0] for item in history], [item[1] for item in history], linewidth=2)
    ax.set(xlabel='Ітерація', ylabel='Log loss', title='Зменшення функції втрат')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_experiments(rows: list[dict], path: Path) -> None:
    fractions = sorted(set(row['train_fraction'] for row in rows))
    values = [max(row['validation_accuracy'] for row in rows if row['train_fraction']==f)
              for f in fractions]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot([int(f*100) for f in fractions], values, marker='o', linewidth=2)
    ax.set(xlabel='Частка доступної тренувальної вибірки, %',
           ylabel='Найкраща точність на валідації',
           title='Вплив обсягу навчальних даних', ylim=(0, 1.02))
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0,1], ['Негативний', 'Позитивний'])
    ax.set_yticks([0,1], ['Негативний', 'Позитивний'])
    ax.set(xlabel='Передбачений клас', ylabel='Фактичний клас', title='Матриця помилок')
    for i in (0,1):
        for j in (0,1):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(dataset: Path, results_dir: Path, seed: int = 42) -> dict:
    texts, labels = load_corpus(dataset)
    # Тестовий набір відокремлений до створення словника, без витоку даних.
    train_texts, test_texts, train_y, test_y = train_test_split(
        texts, labels, test_size=.20, random_state=seed, stratify=labels)
    base_texts, val_texts, base_y, val_y = train_test_split(
        train_texts, train_y, test_size=.25, random_state=seed, stratify=train_y)
    print(f'Дані: {len(texts)} речень, позитивних {sum(labels)}, негативних {len(labels)-sum(labels)}')
    print(f'Навчальні для підбору: {len(base_texts)}, валідаційні: {len(val_texts)}, тестові: {len(test_texts)}')

    experiments = []
    for fraction in (.40, .70, 1.0):
        if fraction == 1.0:
            texts_part, y_part = base_texts, base_y
        else:
            texts_part, _, y_part, _ = train_test_split(
                base_texts, base_y, train_size=fraction, random_state=seed, stratify=base_y)
        freqs, scaler, X_part = prepare(texts_part, y_part)
        X_val = scaler.transform(feature_matrix(val_texts, freqs))
        for alpha in (.01, .05, .10):
            for num_iters in (100, 400, 1200):
                theta, _ = gradient_descent(X_part, y_part, alpha, num_iters,
                                           save_every=num_iters)
                validation = score(X_val, val_y, theta)
                experiments.append(dict(train_fraction=fraction,
                    n_train=len(texts_part), alpha=alpha, iterations=num_iters,
                    validation_accuracy=validation['accuracy'],
                    validation_logloss=validation['loss']))
    # Вибір без підглядання у тест. Для однакової точності використовуємо logloss.
    best = min(experiments, key=lambda row: (-row['validation_accuracy'],
                                               row['validation_logloss'], row['iterations']))
    # Фінальна модель донавчається на ВСЬОМУ train (80%), тест використовується один раз.
    freqs, scaler, X_train = prepare(train_texts, train_y)
    theta, history = gradient_descent(X_train, train_y, best['alpha'],
                                      best['iterations'], save_every=max(1,best['iterations']//100))
    X_test = scaler.transform(feature_matrix(test_texts, freqs))
    result = score(X_test, test_y, theta)
    result.update(dict(corpus='UCI IMDb Sentiment Labelled Sentences' if dataset == DEFAULT_DATA
                       else str(dataset.name), total=len(texts), train=len(train_texts),
                       validation=len(val_texts), test=len(test_texts), seed=seed,
                       weights=[float(w) for w in theta], best_hyperparameters=best,
                       dataset_url=DATASET_URL))

    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    save_csv(results_dir/'hyperparameters.csv',
             ['train_fraction','n_train','alpha','iterations','validation_accuracy','validation_logloss'],
             [[row[key] for key in ['train_fraction','n_train','alpha','iterations','validation_accuracy','validation_logloss']]
              for row in experiments])
    save_csv(results_dir/'training_history.csv', ['iteration', 'train_logloss'], history)
    all_words = sorted(set(word for (word, label) in freqs))
    word_rows = [(w, freqs.get((w,1), 0), freqs.get((w,0), 0)) for w in all_words]
    word_rows.sort(key=lambda x: -(x[1]+x[2]))
    save_csv(results_dir/'word_frequencies.csv', ['stem', 'positive', 'negative'], word_rows)
    probabilities = sigmoid(X_test @ theta)
    predictions = (probabilities >= .5).astype(int)
    save_csv(results_dir/'test_predictions.csv',
             ['text', 'true_label', 'predicted_label', 'probability_positive'],
             [[text, int(label), int(pred), round(float(prob),6)]
              for text, label, pred, prob in zip(test_texts, test_y, predictions, probabilities)])
    plot_history(history, results_dir/'training_loss.png')
    plot_experiments(experiments, results_dir/'training_size_vs_accuracy.png')
    plot_confusion(result['confusion_matrix'], results_dir/'confusion_matrix.png')

    examples = ['I really enjoyed this movie, it was wonderful!',
                'The plot was boring and the acting was terrible.',
                'This film is not good, I would not recommend it.']
    print('\nВласні приклади (англійською):')
    demo_rows=[]
    for example in examples:
        vec = scaler.transform(feature_matrix([example], freqs))
        probability=float(sigmoid(vec @ theta)[0])
        label='Позитивна' if probability >= 0.5 else 'Негативна'
        print(f'{label}: {probability:.4f} — {example}')
        demo_rows.append([example, round(probability,6), label])
    save_csv(results_dir/'own_examples.csv', ['text','probability_positive','result'], demo_rows)
    print(f'\nНайкращі параметри: alpha={best["alpha"]}, ітерацій={best["iterations"]}, частка={best["train_fraction"]}')
    print('Точність на відкладеній тестовій вибірці:', f'{result["accuracy"]*100:.2f}%')
    print('Матриця помилок:', result['confusion_matrix'])
    print('Результати збережено у:', results_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='ЛР №1 — логістична регресія для sentiment analysis')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='TSV файл: текст<TAB>0/1')
    parser.add_argument('--no-download', action='store_true', help='Не завантажувати стандартні дані')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.data.exists():
        if args.no_download or args.data != DEFAULT_DATA:
            parser.error(f'Не знайдено файл: {args.data}')
        download_imdb_dataset(args.data)
    run(args.data, args.results_dir, args.seed)


if __name__ == '__main__':
    main()
