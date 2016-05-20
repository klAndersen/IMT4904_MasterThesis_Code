"""
File for handling training of the classifiers
"""

from constants import PLATFORM_IS_WINDOWS, CPU_COUNT
from file_processing import dump_pickle_model

import numpy
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

CROSS_VALIDATION = StratifiedKFold(n_folds=5)


def __create_default_sgd_pipeline(random_state=int(0)):
    """
    Creates a pipeline with a CountVectorizer, TfidfTransformer and SGD Classifier where all values are set

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


def __create_default_svc_pipeline(predict_proba=True, random_state=int(0)):
    """
    Creates a pipeline with a CountVectorizer, TfidfTransformer and SVC Classifier where all values are set

    Arguments:
        predict_proba (bool): Should probability be calculated (goes slower, but allows to see class probability).
        random_state (int): Value for random_state. 0 = no random state.

    Returns:
        Pipeline: Returns constructed pipeline

    """
    return Pipeline([
        ('vect', CountVectorizer(analyzer='word', stop_words='english', min_df=0.01, max_df=0.95)),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(probability=predict_proba, random_state=random_state)),
    ])


def __create_default_sgd_grid_parameters():
    """
    Creates a dictionary containing parameters to use in GridSearch, where all values are set
    """
    grid_parameters = {
        'vect__min_df': (0.01, 0.025, 0.05, 0.075, 0.1),
        'vect__max_df': (0.25, 0.5, 0.75, 0.95, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l1', 'l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 75, 100),
        # 'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__loss': ['log'],
    }
    return grid_parameters


def __create_default_svc_grid_parameters():
    """
    Creates a dictionary containing parameters to use in GridSearch, where all values are set
    """
    # t C:\Users\KnutLucas\Documents\GitHub\IMT4904_MasterThesis_Code\training_data\ training_data_10000 0
    # c = numpy.logspace(-2, 10, 13)
    # gamma = numpy.logspace(-9, 3, 13)
    # param_svm = [
    #     {'clf__C': c, 'clf__kernel': ['linear']},
    #     {'clf__C': c, 'clf__gamma': gamma, 'clf__kernel': ['rbf']},
    #     {'clf__C': c, 'clf__gamma': gamma, 'clf__kernel': ['sigmoid']},
    # ]
    # c = numpy.logspace(0, 10, 11)
    # gamma = numpy.logspace(-9, -1, 10)
    param_svm = [
        {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear']},
        {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [0.001, 0.0001], 'clf__kernel': ['rbf']},
        {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [0.001, 0.0001], 'clf__kernel': ['sigmoid']},
    ]
    return param_svm


def print_classifier_results(svm_detector):
    print("Best score:", svm_detector.best_score_)
    print("Best parameters:", svm_detector.best_params_)
    print("Best estimator:", svm_detector.best_estimator_)


def create_gridsearch(pipeline, parameters, cv, refit=True, n_jobs=-1, scoring="accuracy"):
    """

    Arguments:
        pipeline (sklearn.pipeline.Pipeline): Estimator for the classifier
        parameters (dict): Dictionary with parameter settings
        cv (sklearn.model_selection._split._BaseKFold): Type of cross-validation to use
        refit (bool): Should data be refitted to match the best classifier?
        n_jobs (int): Number of jobs to run in parallel (amount of cores used). Default: -1 = All
        scoring (str): Score type. Default is accuracy

    Returns:
        GridSearchCV: Created GridSearchCV object to use to train classifier

    """
    # according to this post, verbose doesn't work well with multi-threading on windows:
    # http://stackoverflow.com/questions/28005307/gridsearchcv-no-reporting-on-high-verbosity
    if PLATFORM_IS_WINDOWS and n_jobs == -1:
        n_jobs = CPU_COUNT - 1

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        refit=refit,
        n_jobs=n_jobs,
        scoring=scoring,
        cv=cv,
        verbose=10
    )


def create_and_save_model(train_data, labels, model_path=str, predict_proba=True, test_size=float(0.2), random_state=0,
                          print_results=True, use_sgd_settings=False):
    """
    Creates a classifier model by using ```train_test_split``` to split data and ```GridSearchCV``` to select best fit
    
    Arguments:
        train_data: The training data for the classifier
        labels: Labels for the training data
        model_path (str): The location to store the created model
        test_size (float): Size of test set (amount of training data to use for testing classifier) (default=0.2)
        predict_proba (bool): Should probability be calculated (goes slower, but allows to see class probability).
                            Only for SVC
        random_state (int): Random state
        print_results (bool): Should the results of the classifier be printed?
        use_sgd_settings (bool): If True, run exhaustive SGD search. If False, use exhaustive SVC

    Returns:
        sklearn.model_selection._search.GridSearchCV: The ```GridSearchCV``` classifier model

    """
    # split all the training data into both training and test data
    question_train, question_test, label_train, label_test = train_test_split(train_data,
                                                                              labels,
                                                                              test_size=test_size,
                                                                              random_state=random_state)
    # get setup and create grid
    if use_sgd_settings:
        pipeline_svm = __create_default_sgd_pipeline()
        param_svm = __create_default_sgd_grid_parameters()
    else:
        pipeline_svm = __create_default_svc_pipeline(predict_proba)
        param_svm = __create_default_svc_grid_parameters()
    grid_svm = create_gridsearch(pipeline_svm, param_svm, CROSS_VALIDATION)
    svm_detector = grid_svm.fit(question_train, label_train)
    dump_pickle_model(svm_detector, model_path)
    if print_results:
        print("Classifier results:")
        print_classifier_results(svm_detector)
        if test_size > 0:
            predict_split = svm_detector.predict(question_test)
            print("Confusion matrix for test set classification:")
            print(confusion_matrix(label_test, predict_split))
            print("Accuracy score for test set:")
            print(accuracy_score(label_test, predict_split))
            print("Classification Report:")
            print(classification_report(label_test, predict_split))
    return svm_detector


def create_singular_feature_detector_model(pipeline_svm, param_svm, model_path, train_data, labels,
                                           test_size=float(0.2), random_state=0, print_results=True):
    """
    Create a model based on the best estimator - and parameter values from previously trained model

    Arguments:
        pipeline_svm (sklearn.pipeline.Pipeline): Pipeline containing the best estimator
        param_svm: Dictionary containing the best parameters
        train_data: The training data for the classifier
        labels: Labels for the training data
        model_path (str): The location to store the created model
        test_size (float): Size of test set (amount of training data to use for testing classifier) (default=0.2)
        random_state (int): Random state
        print_results (bool): Should the results of the classifier be printed?

    """
    # split the data set into training and test set
    question_train, question_test, label_train, label_test = train_test_split(train_data,
                                                                              labels,
                                                                              test_size=test_size,
                                                                              random_state=random_state)
    # create a grid search using the set values
    grid_svm = create_gridsearch(pipeline_svm, param_svm, CROSS_VALIDATION)
    svm_detector = grid_svm.fit(question_train, label_train)
    dump_pickle_model(svm_detector, model_path)
    if print_results:
        print("Classifier results:")
        print_classifier_results(svm_detector)
        if test_size > 0:
            predict_split = svm_detector.predict(question_test)
            print("Confusion matrix for test set classification:")
            print(confusion_matrix(label_test, predict_split))
            print("Accuracy score for test set:")
            print(accuracy_score(label_test, predict_split))
            print("Classification Report:")
            print(classification_report(label_test, predict_split))
