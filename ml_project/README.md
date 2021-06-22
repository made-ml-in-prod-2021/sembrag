ml_project
==============================

"ML in production" homework 1.  

Project usage
---------------
To run model train pipeline use:

```> python ml_project\src\full_model_pipeline.py path\to\your\config.yml```

For example for random forest config (watch configs folder):

```> python ml_project\src\full_model_pipeline.py ml_project\configs\config_rf.yml```

To run tests use:

```> pytest --cov=src -v ml_project\tests\```

Самооценка:
----------
0) Назовите ветку homework1 (1 балл) - 1 балл
1) положите код в папку ml_project
2) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения. (2 балла)
1) Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов) - 2 балла
2) Проект имеет модульную структуру(не все в одном файле =) ) (2 баллов) - 2 балла
3) использованы логгеры (2 балла) - 2 балла
4) написаны тесты на отдельные модули и на прогон всего пайплайна(3 баллов) - 3 балла
5) Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов) - 3 балла
6) Обучение модели конфигурируется с помощью конфигов yaml, закомиччены 2 конфигурации (3 балла) - 3 балла

7) Используются датаклассы для сущностей из конфига, а не голые dict (3 балла) - 3 балла 

8) Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла) - 2 балл

9) Обучите модель, запишите в readme как это предлагается (3 балла) - 3 балла

10) напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать (3 балла)  - 1 балл

11) Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы) - 0 баллов

12) Настроен CI(прогон тестов, линтера) на основе github actions  (3 балла - доп баллы) - 0 баллов
13) Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) - 0 баллов

Project Organization
------------


    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model scores
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── config         <- Folder to contain dataclasses for configs
    │   │   │                 
    │   │   ├── config.py  <- Full config 
    │   │   └── data_config.py      <- Datafile configs
    │   │   ├── feature_config.py   <- Feature configs  
    │   │   └── model_config.py     <- Model configs
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
