Advances in Financial Machine Learning Exercises
==============================

Experimental solutions to selected exercises from the book [Advances in Financial Machine Learning by Marcos Lopez De Prado](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482109)

Make sure to use `python setup.py install` in your environment so the `src` scripts which include `bars.py` and `snippets.py` can be found by the jupyter notebooks and other scripts you may develop.

## Additional AFML Projects and Resources
There are other github projects and links that people share that are inspired by the book. I'd like to collect them here to share with others in the spirit of collaboration and idea sharing. If you have more to add please let me know. 

### Github Projects

- [The Open Source Hedge Fund Project](http://www.quantsportal.com/the-open-source-hedge-fund-project/)
	- [Github MLFinLab Repo](https://github.com/hudson-and-thames/mlfinlab)
	- [Github Notebooks](https://github.com/hudson-and-thames/research)

- [rspadim Github](https://github.com/rspadim/Adv_Fin_ML/)

### Article Links

- [Financial Machine Learning Part 0: Bars by Maks Ivanov](https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba)
- [Deflated Sharpe Ratio](https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html) - [Gautier Marti blog](https://gmarti.gitlab.io/)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
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
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
