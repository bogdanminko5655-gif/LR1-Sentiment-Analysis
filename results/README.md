# Результати

Після `python main.py` тут будуть автоматично створені:

- `metrics.json`: перевірені метрики на тестовій вибірці, матриця помилок, вибрані параметри;
- `hyperparameters.csv`: 27 валідаційних експериментів;
- `word_frequencies.csv`: позитивні й негативні частотності стемів;
- `test_predictions.csv`: передбачення для **відкладеної** тестової вибірки;
- `own_examples.csv`: три власні англомовні речення;
- `training_history.csv`, `training_loss.png`, `training_size_vs_accuracy.png`, `confusion_matrix.png`.

**У вихідному архіві немає вигаданих метрик IMDb.** Для справжніх результатів спочатку запустіть `python main.py`, потім `python make_report.py`.
