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
