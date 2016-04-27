"""
Main entry file, all user interaction is handled through this class
"""

# python imports
import nltk
import pickle
from time import time, ctime
from pandas import DataFrame
# scikit-learn imports
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# project imports
from python35_version.mysqldatabase import MySQLDatabase
from constants import CLASS_LABEL_KEY, QUESTION_TEXT_KEY, FILEPATH_TRAINING_DATA, FILEPATH_MODELS, DATABASE_LIMIT

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


def create_default_lsvc_pipeline(random_state=int(0)):
    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='word', stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC(dual=True, penalty='l2', random_state=random_state))
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


def create_default_grid_parameters_lsvc():
    """
    Creates a dictionary containing parameters to use in GridSearch, where all values are set
    """

    # LinearSVC.keys(['max_iter', 'intercept_scaling', 'multi_class', 'loss', 'tol', 'C', 'dual', 'class_weight',
    # 'penalty', 'fit_intercept', 'verbose', 'random_state'])

    grid_parameters = {
        'vect__min_df': (0.01, 0.025, 0.05, 0.075, 0.1),
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': (1, 10, 100, 1000),
        'clf__max_iter': (100, 250, 500, 750, 1000, 1500, 2000),
        'clf__loss': ('hinge', 'squared_hinge'),
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
def load_training_data(file_location=str, load_from_database=False, limit=int(1000), clean_dataset=True):
    """
    If ```load_from_database``` is True, retrieves and stores data from database to file.

    Arguments:
        file_location (str): Path + filename of libsvm file to save/load (e.g. 'training_data')
        load_from_database (bool): Should data be retrieved from database?
        limit (int): Amount of records to retrieve from database (default=1000)
        clean_dataset (bool): Should questions be cleaned (e.g. remove code samples, hexadecimals, numbers, etc)?

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
        data = MySQLDatabase().retrieve_training_data(limit, clean_dataset)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
        td_matrix = vectorizer.fit_transform(data.get(QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[CLASS_LABEL_KEY], f=svm_file, comment=comment)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)




def split_training_data(dataset, labels, test_size=float(0.2), random_state=int(0)):
    # split all the training data into both training and test data (test data = 20%)
    return train_test_split(dataset,
                            labels,
                            test_size=test_size,
                            random_state=random_state)


path = "./test_all_alg_pickle/"

# good question, ID: 927358
good_question = "I committed the wrong files to Git. How can I undo this commit?"
# bad question, ID: 27391628
bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
               "a header that provides the link and a manual containing instructions on how to use the library. " \
               "How can you access the private data member of the class? Note this is only specific to C++. " \
               "Normally there's no way you can access a private data member other than making " \
               "friends or writing a getter function, both of which require changing the interface of the " \
               "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
               "This is not a problem from a textbook or class assignment."

time_start = time()
current_time(time_start, "Program started")
# retrieve data to use
db_limit = DATABASE_LIMIT.get('10000')
filename = FILEPATH_TRAINING_DATA + str(db_limit)
so_dataframe, (training_data, class_labels) = load_training_data(filename, False, db_limit, True)

_unprocessed_filename = FILEPATH_TRAINING_DATA + str(db_limit) + "_unprocessed"
_unprocessed_so_dataframe, (training_data, class_labels) = load_training_data(filename, True, db_limit, False)


corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]

# stem and update the data in the pandas.dataframe
counter = 0
for question in corpus:
    corpus[counter] = stem_training_data(question)
    counter += 1


s_question_train, s_question_test, s_label_train, s_label_test = split_training_data(corpus,
                                                                             so_dataframe[CLASS_LABEL_KEY])

stem_good_question = stem_training_data(good_question)
stem_bad_question = stem_training_data(bad_question)


u_question_train, u_question_test, u_label_train, u_label_test = split_training_data(_unprocessed_so_dataframe[QUESTION_TEXT_KEY],
                                                                                     _unprocessed_so_dataframe[CLASS_LABEL_KEY])

question_train, question_test, label_train, label_test = split_training_data(so_dataframe[QUESTION_TEXT_KEY],
                                                                             so_dataframe[CLASS_LABEL_KEY])


def create_pipeline(vectorizer, classifier):
    return Pipeline([
        ('bow', vectorizer),
        ('tfidf', TfidfTransformer()),
        ('classifier', classifier),
    ])

# , 10000, 100000; 10^4 & 10^5

# pipeline parameters to automatically explore and tune
param_svm = [
    # Values for the classifier (C & LINEAR kernel)
    {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
    # Values for the classifier (C, Gamma & RBF kernel)
    {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
    # Values for the classifier (C, SIGMOID kernel)
    {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['sigmoid']},
]


def create_gridsearch(pipeline, parameters, cv):
    return GridSearchCV(
        pipeline,  # pipeline from above
        param_grid=parameters,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=cv,  # what type of cross validation to use
        verbose=1
    )


def print_accuracy_scores(svm_detector, test_l, test_q, good_q, bad_q):
    print(svm_detector.grid_scores_)
    print(svm_detector.predict([good_q])[0])
    print(svm_detector.predict([bad_q])[0])
    print(confusion_matrix(test_l, svm_detector.predict(test_q)))
    print(classification_report(test_l, svm_detector.predict(test_q)))
    print(svm_detector.best_score_)
    print(svm_detector.best_params_)
    print(svm_detector.best_estimator_)

cv = StratifiedKFold(n_folds=5)
cv_pipeline = create_pipeline(CountVectorizer(analyzer='word', min_df=1), SVC())
tfid_pipeline = create_pipeline(TfidfVectorizer(analyzer='word', min_df=1), SVC())

cv_pipeline2 = create_pipeline(CountVectorizer(analyzer='word', stop_words='english', min_df=0.01), SVC())
tfid_pipeline2 = create_pipeline(TfidfVectorizer(analyzer='word', stop_words='english', min_df=0.01), SVC())

try:
    time_start = time()
    current_time(time_start, "Unprocessed SVC")
    grid_svm = create_gridsearch(cv_pipeline, param_svm, cv)
    u_cv_svm_detector = grid_svm.fit(u_question_train, u_label_train)

    grid_svm = create_gridsearch(tfid_pipeline, param_svm, cv)
    u_tfid_svm_detector = grid_svm.fit(u_question_train, u_label_train)

    model_name = path + "u_cv_svm_detector_svc"
    dump_pickle_model(u_cv_svm_detector, model_name)
    model_name = path + "u_tfid_svm_detector_svc"
    dump_pickle_model(u_tfid_svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(u_cv_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    print('\n')
    print('\n')
    print_accuracy_scores(u_tfid_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)

try:
    time_start = time()
    current_time(time_start, "Unprocessed SVC2")
    grid_svm = create_gridsearch(cv_pipeline2, param_svm, cv)
    u_cv_svm_detector = grid_svm.fit(u_question_train, u_label_train)

    grid_svm = create_gridsearch(tfid_pipeline2, param_svm, cv)
    u_tfid_svm_detector = grid_svm.fit(u_question_train, u_label_train)

    model_name = path + "u_cv_svm_detector_svc_stopwords"
    dump_pickle_model(u_cv_svm_detector, model_name)
    model_name = path + "u_tfid_svm_detector_svc_stopwords"
    dump_pickle_model(u_tfid_svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(u_cv_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    print('\n')
    print('\n')
    print_accuracy_scores(u_tfid_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)


try:
    current_time(time_start, "Normal SVC")
    grid_svm = create_gridsearch(cv_pipeline, param_svm, cv)
    cv_svm_detector = grid_svm.fit(question_train, label_train)

    grid_svm = create_gridsearch(tfid_pipeline, param_svm, cv)
    tfid_svm_detector = grid_svm.fit(question_train, label_train)

    model_name = path + "cv_svm_detector_svc"
    dump_pickle_model(cv_svm_detector, model_name)
    model_name = path + "tfid_svm_detector_svc"
    dump_pickle_model(tfid_svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(cv_svm_detector, label_test, question_test, good_question, bad_question)
    print('\n')
    print('\n')
    print_accuracy_scores(tfid_svm_detector, label_test, question_test, good_question, bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)


try:
    current_time(time_start, "Normal SGD")
    # get setup and create grid
    pipeline_svm = create_default_sgd_pipeline()
    param_svm = create_default_grid_parameters()
    grid_svm = GridSearchCV(pipeline_svm, param_grid=param_svm, n_jobs=-1, verbose=1, cv=cv)

    u_svm_detector = grid_svm.fit(u_question_train, u_label_train)
    svm_detector = grid_svm.fit(question_train, label_train)

    model_name = path + "u_svm_detector_sgd"
    dump_pickle_model(u_svm_detector, model_name)
    model_name = path + "svm_detector_sgd"
    dump_pickle_model(svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(u_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    print('\n')
    print('\n')
    print_accuracy_scores(svm_detector, label_test, question_test, good_question, bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)

try:
    current_time(time_start, "Stemmed SVC")
    grid_svm = create_gridsearch(cv_pipeline, param_svm, cv)
    cv_svm_detector = grid_svm.fit(s_question_train, s_label_train)

    grid_svm = create_gridsearch(tfid_pipeline, param_svm, cv)
    tfid_svm_detector = grid_svm.fit(s_question_train, s_label_train)

    model_name = path + "s_cv_svm_detector_svc"
    dump_pickle_model(cv_svm_detector, model_name)
    model_name = path + "s_tfid_svm_detector_svc"
    dump_pickle_model(tfid_svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(cv_svm_detector, s_label_test, s_question_test, stem_good_question, stem_bad_question)
    print('\n')
    print('\n')
    print_accuracy_scores(tfid_svm_detector, s_label_test, s_question_test, stem_good_question, stem_bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)


try:
    current_time(time_start, "Stemmed SGD")
    # get setup and create grid
    pipeline_svm = create_default_sgd_pipeline()
    param_svm = create_default_grid_parameters()
    grid_svm = GridSearchCV(pipeline_svm, param_grid=param_svm, n_jobs=-1, verbose=1, cv=cv)

    svm_detector = grid_svm.fit(s_question_train, s_label_train)

    model_name = path + "s_svm_detector_sgd"
    dump_pickle_model(svm_detector, model_name)

    # print results in this round
    print_accuracy_scores(svm_detector, s_label_test, s_question_test, stem_good_question, stem_bad_question)
    print('\n')
    print('\n')
except Exception as ex:
    print(ex)

try:
    current_time(time_start, "LinearSVC")
    # get setup and create grid
    pipeline_svm = create_default_lsvc_pipeline()
    param_svm = create_default_grid_parameters_lsvc()
    grid_svm = GridSearchCV(pipeline_svm, param_grid=param_svm, n_jobs=-1, verbose=1, cv=cv)

    u_svm_detector = grid_svm.fit(u_question_train, u_label_train)
    model_name = path + "u_svm_detector_lsvc"
    dump_pickle_model(u_svm_detector, model_name)

    svm_detector = grid_svm.fit(question_train, label_train)
    model_name = path + "svm_detector_lsvc"
    dump_pickle_model(svm_detector, model_name)

    stem_svm_detector = grid_svm.fit(s_question_train, s_label_train)
    model_name = path + "s_svm_detector_lsvc"
    dump_pickle_model(stem_svm_detector, model_name)

    # print results in this round
    print(u_svm_detector)
    print('\n')
    print('\n')
    print(svm_detector)
    print('\n')
    print('\n')
    print(stem_svm_detector)
    print('\n')
    print('\n')

    print("Attempting multi-print of result u_svm_detector")

    print('\n')
    print(u_svm_detector.best_score_)
    print(u_svm_detector.best_params_)
    print(u_svm_detector.best_estimator_)
    print('\n')
    print('\n')

    print("Attempting multi-print of result svm_detector")

    print('\n')
    print(svm_detector.best_score_)
    print(svm_detector.best_params_)
    print(svm_detector.best_estimator_)
    print('\n')
    print('\n')

    print("Attempting multi-print of result u_svm_detector")

    print('\n')
    print(stem_svm_detector.best_score_)
    print(stem_svm_detector.best_params_)
    print(stem_svm_detector.best_estimator_)
    print('\n')
    print('\n')

    # print results in this round
    print_accuracy_scores(u_svm_detector, u_label_test, u_question_test, good_question, bad_question)
    # print results in this round
    print_accuracy_scores(svm_detector, label_test, question_test, good_question, bad_question)
    # print results in this round
    print_accuracy_scores(stem_svm_detector, s_label_test, s_question_test, stem_good_question, stem_bad_question)
except Exception as ex:
    print(ex)
