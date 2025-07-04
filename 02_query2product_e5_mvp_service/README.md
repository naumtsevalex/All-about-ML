# 🔎 Поиск по товарам на эмбеддингах E5

## Описание

Этот проект — сервис интеллектуального поиска по товарам с помощью современных языковых моделей (E5). 
Пользователь вводит текстовый запрос, а система находит наиболее релевантные товары из большой базы, используя нейросетевые эмбеддинги.

- Используется датасет [tasksource/esci](https://huggingface.co/datasets/tasksource/esci) (запросы, товары, метки релевантности).
- Модель дообучается на задаче поиска (Triplet Loss).
- Для всех товаров считаются эмбеддинги.
- Веб-интерфейс реализован на Streamlit.

---

## Структура проекта

```
├── app.py                    # Streamlit веб-приложение для поиска
├── EDA_plus_baseline.ipynb   # EDA и анализ данных
├── train_e5.ipynb            # Обучение и дообучение модели E5
├── calc_all_embed.ipynb      # Расчет эмбеддингов для товаров
├── product_titles.csv        # Названия товаров
├── product_embeddings.npy    # Эмбеддинги товаров (большой файл)
├── product_stats.csv         # Статистика просмотров товаров
├── saved_e5_model/           # Сохраненная модель
├── checkpoints/              # Чекпоинты обучения
├── runs/                     # Логи и результаты экспериментов
├── logs/                     # Логи (например, для TensorBoard)
├── images/                   # Картинки для README (создать вручную)
└── README.md                 # Этот файл
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
# или вручную:
pip install streamlit torch transformers pandas numpy matplotlib seaborn tqdm datasets
```

### 2. EDA и подготовка данных

- Открой `EDA_plus_baseline.ipynb` для анализа и первичной подготовки данных.
- Используй Jupyter или VSCode.

### 3. Обучение модели

- Запусти `train_e5.ipynb` для дообучения модели E5 на своих данных.
- Чекпоинты сохраняются в папку `checkpoints/`.

### 4. Расчет эмбеддингов товаров

- Запусти `calc_all_embed.ipynb` для получения эмбеддингов всех товаров.
- Результат сохраняется в `product_embeddings.npy`.

### 5. Запуск веб-сервиса поиска

```bash
streamlit run app.py
```
- Откроется веб-интерфейс для поиска по товарам.

---

## Внешние сервисы и мониторинг

### TensorBoard (если есть логи обучения)

```bash
tensorboard --logdir=logs/ --port=6006
```
- Откроется по адресу: http://localhost:6006

### Мониторинг GPU

```bash
nvitop
# или
watch -n 1 nvidia-smi
```

---

## Работа с фоновыми процессами через tmux

Для удобного запуска сервисов (Streamlit, TensorBoard, мониторинг и т.д.) в фоне рекомендуется использовать [tmux](https://github.com/tmux/tmux):

```bash
# Запустить новую сессию tmux с именем mysession
tmux new -s mysession

# Внутри tmux можно запускать любые команды, например:
streamlit run app.py
tensorboard --logdir=logs/ --port=6006
nvitop

# Чтобы отсоединиться от сессии (оставив процессы работать):
Ctrl+b, затем d

# Чтобы вернуться к сессии:
tmux attach -t mysession

# Список всех сессий:
tmux ls

# Завершить сессию:
exit
```

---

## Как приложить картинки

1. Создай папку `images/` в корне проекта.
2. Сохрани туда графики обучения, примеры выдачи, скриншоты интерфейса и т.д.
3. Вставляй картинки в README так:

```markdown
![График обучения](images/train_loss.png)
![Пример выдачи](images/search_example.png)
```

---

## .gitignore

В репозиторий не попадают большие и временные файлы:
```
product_embeddings.npy
small_product_embeddings.npy
product_stats.csv
product_titles.csv
runs/
saved_e5_model/
checkpoints/
logs/
```

---

## Аппаратные требования
- Желательно наличие GPU (CUDA) для обучения и инференса.
- Для запуска Streamlit и EDA достаточно CPU.

---

## Контакты и поддержка
Если возникли вопросы — пиши!
