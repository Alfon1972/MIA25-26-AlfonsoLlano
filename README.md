# dslab-nlp-dl
PC: NATURAL LANGUAGE PROCESSING
MBIT - MIA MAR 25

# Pre Requisites

Create a virtual environment
```bash
python -m venv .venv
```

Activate the Virtual Environment,
following the specific instructions for your OS,
for example, with Win/Powershell: `.\.venv\Scripts\activate`


Install dependencies
```bash
pip install -r requirements.txt
```

This virtual environment has been built using Win11 and Python 3.10.9, 3.11 and 3.12 

As project is based in notebooks, three popular approaches can be followed:
* Install jupyter notebook: `pip install jupyter`
* Install jupyterlab: `pip install jupyterlab`
* Use VSCode notebooks extension: [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) 
and install ipykernel with `pip install ipykernel`


Download `spaCy` models:
```bash
python -m spacy download en_core_news_sm
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg
```
Download `nltk` models:
```bash
python install_nltk_models.py
```

**Important!!**
Create the following folders:
```
models/
 prod/
 archive/

data/
 interim/
 processed/
```

Check the `00_check_setup.ipynb` if everything is running


# Model lifecycle
This project proposes the following cycle:
1. Experimentation: train several algorithms, registering a comparable cross validation metric (CV metric)
2. Benchmark: Compare each experiment using the CV metric and select a `champion model`
3. Test and register: Optionally retrain the champion model and test it. Save the required artifacts to the appropiate `model registry`
4. Develop the software (scripts or modules) to retrain and obtain predictions

After these steps, the model is ready to be put in production, for example, building a REST API Endpoint

In this sample project, the `model registry` is implemented in local folder called `models/`.
It should have the following folders:  

`prod`: Stores the artifacts of the current `champion model` **only**, used to deploy.
For example:
```
model.pickle
metadata.pickle
```

`archive`: Use a versioning naming convention to name subfolders, for example, `YYYYMM` and move each model in `prod` prior to save a new one:
```
202506/
    model.pickle
    metadata.pickle
202507/
    model.pickle
    metadata.pickle
```

Usually, the convention name to subfolders is assigned to the `model_version_id` metadata. So our `model_version_id` is `YYYYMM`


# Project Structure
This project loosely follows Cookiecutter data science standard in order to 
provide a familiar and repeatable structure.

## Data folder (data/)
Contains data assets related to the project. It is divided in 3 folders
* raw/: Input and auxiliary data assets
* interim/: Input dataset with standard column naming (`prep.csv`) with Train and Test subsets (`train.csv` and `test.csv`) for reproducible experiments and experiments results (.e.g `exp01_hpt_nb/df_exp_summary.csv`)
* processed/: Predictions (`scoring_YYYYMM.csv`)


## Models Folder: Model Registry (models/)
Stores the artifacts (like `model.pickle`, plots, tables, and so on) created after 
training and evaluating a champion model

## Source code folder (src/)
Python packages and modules needed in the project.

## Tests (tests/)
Unit testing of the source code

## Scripts
Scripts should be placed in project root
The project should contain a `train.py` script that implements
the code needed to automate the training  of the `champion model` architecture (algorithm + hiperparameters)
(by repeating the proceess followed in experimentation and benchmarking) and storing the resulting artifacts in `models/` as stated.

In addition, regarding the deployment of the `champion model` there should be one 
oj the following alternatives:
* A `score.py` script to run (in batch) predictions from the `models/prod/` artifacts (Compulsory) 
* A script to run a REST API HTTP Server to obtain predictions in real time (Optinal)

## Notebooks
Noteboks can be placed in project root, as long as there are only a handfull of 
them. 


## Model Card
A model card is a document that describes your model in a scientific way,
so that any other colleage can easilly use your work.

A model card is not the technical documentation of a model,
although can be a part of it.

Many model cards and though to be used with Langauge Models, however, 
with predictive models, some tweaks maybe recommended


# Task Description
Build a binary classification model (0/1) by following the next steps:
1. Load data (0.5 pts)
2. Split Data (0.5 pts)
3. Perform an Exploratory Data Analysis (1 pts)
4. Perform Hiper Hyperparameter tunning for a NaiveBayes model (1 pts)
5. Perform Hiper Hyperparameter tunning for a GBT model (1 pts)
6. Benchmark every model candidate and choose a `champion model` (1 pts)
7. Implement source code (1 pts)
8. Write unit tests (1 pts)
9. Develop train.py script and train a model and generate scoring artifacts (1 pts)
10. Develop score.py script and get prediction from production model artifacts (1 pts)
11. Write a Model Card (1 pts)

## Load data
Proposed dataset for this project is from HuggingFace Dataset
 [FR_NFR_Spanish_requirements_classification](https://huggingface.co/datasets/MariaIsabel/FR_NFR_Spanish_requirements_classification)

The goal is to determine whether a ticket for a software product feature is 
Functional (F) or Not Functional (NF), so it can be automatically derived to 
a task to the corresponding team.

Instructions (01_load_and_split.ipynb):
1. Load the dataset as pandas
2. Obtain the number or rows and columns
3. Calculate the percentage of N and NF rows
4. Create a new column named: `y_is_nf` that maps F to 0 and NF to 1
5. Create a new column with the documents' text named: `x_text`


## Split Data
Create a dataset split in train and test subsets. 
Specify a convenient testing size and make sure that it follows best practices 
regarding reproducibility and sampling

Instructions (01_load_and_split.ipynb):
1. Create parametrized constants: `RND_SEED` and `PCT_TEST`
2. Split the data in Train and Test subets: `df_train`, `df_test`
3. Write test and train dataset to interim/
3. Check that the train and test subsets have the expected size and also, 
considering the test subset the expected proportion of 1's

## Perform an Exploratory Data Analysis
This step is crucial to check assumptions on the task at hands and also, 
to be able to define a proper experimentation phase. Regarding this last idea,
the main challenge in a NLP problem is to design a proper vocabulary.

Instructions (02_eda.ipynb):
1. Check the assumption that every text in the dataset is in Spanish
with [langdetect](https://pypi.org/project/langdetect/)
2. Use [sklearn Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to obtain the frequency of each term in the in the dataset (both train and test). You should:
    * Standardize terms by using a Lemmatizer
    * Do NOT remove stopwords
    * Use CountVectorizer to compute  unigrams `ngram_range=(1,1)` and do not exclude any term from the vocabulary, setting appropiatelly: `max_df` and `min_df` (hint: check default values in documentation link) and obtain the DTM matrix (rows -> docs, cols -> terms)
    * Sum DTM matrix by cols so as to get a pandas series where the index is the term and the value is the frequency
3. Get 10 most common and least common terms in the vocabulary.
4. Save the `Top 30 most frequent terms` and `Vocabulary frequency: Bottom ` plot showing top 30 most frequent terms in `output/top_most_freq_terms.png`
5. Save the `Bottom 30 least frequent terms with freq g10` plot showing top 30 least frequent terms with a frequency larger than 10 counts in `output/bottom_least_freq_terms.png`

For both use a vertical plot bar with proper ordering, compare with provided results in `outputs/sample/`. This should give some hints on how to to set `max_df` and `min_df` for a proper model building experiment


##  Perform Hiper Hyperparameter tunning for a NaiveBayes model
This is a simple experiment where a basic model and some feature engineering configurations are fit and cross validated. Use scikit learn to train a NaiveBayes model to be used as baseline.

This Hyperparamenter Tunning campaign (HPT) should be focus on vocabulary size.
This works by changing the Hyperparameters of the pipeline and obtaining  CV-metric for each
configuration. The process is already implemented on sklearn in the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object.


Instructions (03_exp_hpt_naivebayes.ipynb):
1. Build a [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) that fits a XXXVectorizer and a NaiveBayes model. Consider the most appropiate method to build the DTM matrix. Use stemming to standardize terms and unigrams only.
2. Use K-Fold Cross Validation to fit the pipeline and get CV metrics. Store the metrics in interim/. 
**USE ONLY TRAIN SUBSET**


## Perform Hiper Hyperparameter tunning for a GBT model
This is actually a set of experiments consisting on a Hyperparamenter Tunning campaign (HPT)
This works by changing the Hyperparameters of the pipeline and obtaining  CV-metric for each
configuration. The process is already implemented on sklearn in the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object.
This will run a CV job on every configuration defined by the grid.

The result is that a set of models is fit and validated, so it is easy to compare to other
experiments

Instructions (04_exp_hpt_gbt.ipynb):
1. Build a sklearn Pipeline that fits a [GBT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
* Use Stemming for term standartization.
* Use early stopping with GBT
2. GridSearchCV to run the set of experiments and get the summary, store it in `interim/exp02_hpt_gbt/df_exp_summary.csv`.
**USE ONLY TRAIN SUBSET**

(Optionally, build a HGBT)


## Benchmark every model candidate and choose a `champion model`
Read CV metrics from interim for every experiment and select a `champion model` 
based on your criterion.

Instructions (05_benchmark.ipynb):
1. Read interim/ results to get CV metrics from every model
2. Select a `champion model`


## Implement source code

**Congrats!** At this point you almost have the software you need to finish the project. 
The task is to refactor it to make a proper software project, this task will
be completed by developing `src/models.py`:
* In this Python module, you will move (and hopefully refine) the code
 in the notebooks to build the model, especifically:
* Stemming `tokenizer_stemmer_es`
* A function to get the stopwords already tokenized
* `get_model`: should return a sklearn pipeline with the `champion model` architecture


## Write unit tests
Unit testing is a fundamental part of software developing. In building a model,
it will help in the following ways:
* Debug the data prepraration pipeline
* Debug the pipeline built

We are going to ilustrate few simple tests, but feel free to add some more complex ones:
* Testing transformations
* Testing the tokenizer extensivelly
* Test model architecture: Does it converge? Can in actually learn?

Go to tests/test_models.py and finish the tests implementation.
To run the tests, just launch:

```bash
pytest tests/test_models.py -v
```


## Develop train.py script and train a model and generate scoring artifacts

Implement a coherent training pipeline, consider that:
* Your model should be implemented in  `src/models.py`
* Model hiperparameters should be scritps argumments
* takes input dataset from `data/raw`, splits the data in train and test, fits the `champion model` and evaluate it in test data. Finally, archives prior model version and saves current one to `models/prod`.


This script will retrain the model to store artifacts in `models/prod`  with the hyperparameters decided previously

Customize your script arguments for your champion model hiperparameter:

```python
    # TODO: Add your arguments
    parser.add_argument("--min_df", type=int, default=3, help="Min document frequency for vectorizer.")
```

This script will take as input:
* Raw data 
* model version ID to be stored (e.g. 202602)

```python
    parser.add_argument("path_data", type=str, help="Path to the raw dataset CSV file.")
    parser.add_argument("model_version_id", type=str, help="Model version ID (e.g. 202602).")
```

To run the script:
```bash
python train.py ./data/raw/New%20Spanish%20Academic%20Dataset.csv ${VERSION_ID} --min_df ${MIN_DF} --max_df ${MAX_DF}
```

For example:
```bash
python train.py ./data/raw/New%20Spanish%20Academic%20Dataset.csv 202602 --min_df 5
```

**Remember**: Copy the current artifacts from `models/prod` to the corresponding models/archive/`model_version_id`/.
Resulting in something like:
```
models/
    archive/
        202602/
            metadata.pkl
            model.pkl
```

## Develop score.py script and get prediction from production model artifacts
Check that: In `models/prod` there are the required artifacts to build the model,
 and only the current champion model ones.

1. Write a `score.py` script to run batch predictions on file 
and store the results in processed with the model in `models/prod`.
2. Run a scoring job with the model built in 05_benchmark.ipynb



This script will take as input:
* Scoring raw data 
* Prediction file name

```python
    parser.add_argument("path_data", type=str,
                        help="Path to the input data, e.g. ./data/raw/my_file.csv")
    parser.add_argument("output_file", type=str,
                        help="Scored file name, e.g. scoring_YYYYMM.csv")
```

```bash
python score.py ./data/raw/New%20Spanish%20Academic%20Dataset.csv scoring_202602.csv
```

**Optional:** Develop a FastAPI prediction endpoint

## Write a Model Card

Go over the model proposed by [Google Model Cards](https://modelcards.withgoogle.com/explore-a-model-card#model-summary-section)
and write a simple MODEL_CARD.md

For the  Model Card task, analyze:
* Which terms are more important for the model and whether they make senses
* Is the model robust for other languages?


# Final checks and autoevaluation

The final outputs with notebook `09_check_outputs.ipynb`. Then 



# References 
* [How to Set Up a Virtual Environment in Python – And Why It's Useful](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)
* [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
* [PyTest for Data Science](https://codecut.ai/pytest-for-data-scientists/)
* [Argparse Tutorial](https://docs.python.org/3/howto/argparse.html)