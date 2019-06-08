# Neuron: the E-Tools' AI

Conjunto de técnicas de inteligência artificial para o sistema E-Tools viabilizadas por meio de _end-points_.

## routes.py

Responsável por subir o servidor Flask e criar todas as rotas necessárias.

### Rotas

#### /api/classifier/failcode

Método(s): POST.

Recebe um JSON da rota **/classifier/failcode** do Node.js com a seguinte estrutura:

```
{
    "excel":
    {
        "PROBLEMDESCRIPTION":[

        ],
        "FAILCODEDESCRIPTION":[

         ]
    }
}
```

Assim, tem-se os dados para dar início ao procedimento da classificação invocando a função **init_classifier()** contida no arquivo _failcode_classifier.py_

Com a resposta da classificação, retorna-se tais dados de volta ao Node.js.


## Arquivos

### failcode_classifier.py

Tem como objetivo a realização de uma classificação multiclasse a partir dos dados de entrada. Ao fim, retorna um JSON com os resultados da classificação.

**Função inicializadora**: init_classifier(data);

**Parâmetro(s) da função inicializadora**: array com os dados a serem classificados;

**Retorno**: um JSON com os resultados da classificação em que para cada resultado da classificação informa-se também os valores de amostra e acurácia do modelo para aquele item classificado;

**Técnica de Machine Learning utilizada**: classificação multiclasse;

**Algoritmo de Machine-Learning utilizado**: estratégia One-Vs-All (One-vs-the-Rest) com o estimador LinearSVC (Support Vector Classification).


## Uso

Para uso **local** utilize o comando:

Suba o servidor Flask:

```
python routes.py
```


## Links

* [Scikit-learn: One-vs-All](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)

* [Scikit-learn: Linear SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

* [Multiclass Classification: One-vs-all (Coursera - Machine Learning, Stanford University)](https://www.coursera.org/learn/machine-learning/lecture/68Pol/multiclass-classification-one-vs-all)

* [Scikit-learn Pipeline Persistence and JSON Serialization](https://cmry.github.io/notes/serialize)

* [Scikit-learn Pipeline Persistence and JSON Serialization Part II](https://cmry.github.io/notes/serialize-sk)

* [model persistence without pickles](http://thiagomarzagao.com/2015/12/07/model-persistence-without-pickles/)


##  Project Organization
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