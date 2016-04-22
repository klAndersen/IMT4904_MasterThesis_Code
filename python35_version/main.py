"""
Main entry file, all user interaction is handled through this class
"""

import nltk
import pickle
from time import time, ctime
from pandas import DataFrame

from python35_version.mysqldatabase import MySQLDatabase
from constants import CLASS_LABEL_KEY, QUESTION_TEXT_KEY, FILEPATH_TRAINING_DATA, FILEPATH_MODELS, DATABASE_LIMIT

from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


label_test = None
question_test = None
svm_detector_all = None
svm_detector_split = None
mem = Memory("./mem_cache")


def load_pickle_model(file_name=str):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model


def dump_pickle_model(data, file_name=str):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def create_default_sgd_pipeline(random_state=int(0)):
    """
    Creates a pipeline with a CountVectorizer, TfidfTransformer and SGDClassifier where all values are set

    Arguments:
        random_state (int): Value for random_state. 0 = no random state.

    Returns:
        Pipeline: Returns constructed pipeline

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='word', stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=random_state))
    ])
    return pipeline


def create_default_grid_parameters():
    """
    Creates a dictionary containing parameters to use in GridSearch, where all values are set
    """
    grid_parameters = {
        'vect__min_df': (0.01, 0.025, 0.05, 0.075, 0.1),
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l1', 'l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 75, 100),
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
    }
    return grid_parameters


def current_time(time_now, info=str):
    """
    Prints the current time (Date HH:mm:ss) by passing ```time_now``` to ctime()

    Arguments:
        time_now (time&#40;&#41;): The time when this function was called
        info (str): Optional string containing info about this time
    """
    time_now = ctime(time_now)
    if info is None:
        print(time_now)
    else:
        print(info, time_now)
    print('\n')


def stem_training_data(stemming_data=str):
    """
    Removes affixes and returns the stem

    Arguments:
        stemming_data (str): Data to stem

    Returns:
        str: String containing the stemmed data.
        E.g. the words 'cry, 'crying', 'cried' would all return 'cry'.

    """
    porter = nltk.PorterStemmer()
    stemming_data = stemming_data.lower().split()
    m2 = map(lambda x: porter.stem(x), stemming_data)
    return ' '.join(m2)


@mem.cache
def load_training_data(file_location=str, load_from_database=False, limit=1000):
    """
    If ```load_from_database``` is True, retrieves and stores data from database to file.

    Arguments:
        file_location (str): Path + filename of libsvm file to save/load (e.g. 'training_data')
        load_from_database (bool): Should data be retrieved from database?
        limit (int): Amount of records to retrieve from database (default=1000)

    Returns:
         (pandas.DataFrame.from_csv, sklearn.datasets.load_svmlight_file):
         Tuple containing a pandas.DataFrame (all data retrieved from database) and
         tuple with training data (load_svmlight_file)

    See:
        | ```MySQLDatabase().retrieve_training_data```
        | ```pandas.DataFrame.to_csv```
        | ```pandas.DataFrame.from_csv```
        | ```sklearn.datasets.dump_svmlight_file```
        | ```sklearn.datasets.load_svmlight_file```
    """
    svm_file = file_location + ".dat"
    csv_file = file_location + ".csv"
    if load_from_database:
        comment = u"label: (-1: Bad question, +1: Good question); features: (term_id, frequency)"
        MySQLDatabase().set_vote_value_params()
        data = MySQLDatabase().retrieve_training_data(limit)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
        td_matrix = vectorizer.fit_transform(data.get(QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[CLASS_LABEL_KEY], f=svm_file, comment=comment)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)


time_start = time()
current_time(time_start, "Program started")

db_limit = DATABASE_LIMIT.get('10000')
filename = FILEPATH_TRAINING_DATA + str(db_limit)
so_dataframe, (training_data, class_labels) = load_training_data(filename, False, db_limit)

counter = 0
for question in so_dataframe[QUESTION_TEXT_KEY]:
    so_dataframe.loc[counter, QUESTION_TEXT_KEY] = stem_training_data(question)
    counter += 1

corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]

pickle_exists = False
create_model_from_all_data = True

if __name__ == "__main__":
    if pickle_exists:
        # set paths and load model(s)
        mod_all_data_path = FILEPATH_MODELS + 'svm_detector_all.pkl'
        mod_split_data_path = FILEPATH_MODELS + 'svm_detector_split.pkl'
        svm_detector_all = load_pickle_model(mod_all_data_path)
        svm_detector_split = load_pickle_model(mod_split_data_path)
    else:
        # get setup and create grid
        pipeline_svm = create_default_sgd_pipeline()
        param_svm = create_default_grid_parameters()
        grid_svm = GridSearchCV(pipeline_svm, param_grid=param_svm, n_jobs=-1, verbose=1)
        # start timer and create model
        time_start = time()
        current_time(time_start, "Starting SVM & GridSearch")
        # what data should the model be based on?
        if create_model_from_all_data:
            svm_detector_all = grid_svm.fit(corpus, class_labels)
        else:
            # split all the training data into both training and test data (test data = 20%)
            question_train, question_test, label_train, label_test = train_test_split(corpus,
                                                                                      class_labels,
                                                                                      test_size=0.2,
                                                                                      random_state=0)
            svm_detector_split = grid_svm.fit(question_train, label_train)
        # set end time
        time_start = time()
        current_time(time_start, "Finished")

    if create_model_from_all_data:
        print('\n')
        print(svm_detector_all.best_score_)
        print(svm_detector_all.best_params_)
        print(svm_detector_all.best_estimator_)
        print('\n')
    else:
        print('\n')
        print(svm_detector_split.best_score_)
        print(svm_detector_split.best_params_)
        print(svm_detector_split.best_estimator_)
        print('\n')
