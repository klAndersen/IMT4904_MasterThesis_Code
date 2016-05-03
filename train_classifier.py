"""
File for handling training of the classifiers
"""

import text_processor
from file_processing import load_classifier_model_and_dataframe, dump_pickle_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from constants import DATABASE_LIMIT, FILEPATH_TRAINING_DATA, FILEPATH_MODELS, QUESTION_TEXT_KEY, CLASS_LABEL_KEY


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


def print_classifier_results(svm_detector):
    print("Best score:", svm_detector.best_score_)
    print("Best parameters:", svm_detector.best_params_)
    print("Best estimator:", svm_detector.best_estimator_)


def create_and_save_model(train_data, labels, model_path=str, test_size=float(0.2), random_state=0,
                          print_results=True, use_default_settings=True):
    """
    Creates a classifier model by using ```train_test_split``` to split data and gridsearch to select best fit
    
    Arguments:
        train_data: The training data for the classifier
        labels: Labels for the training data
        model_path (str): The location to store the created model
        test_size (float): Size of test set (amount of training data to use for testing classifier) (default=0.2)
        random_state (int): Random state
        print_results (bool): Should the results of the classifier be printed?
        use_default_settings (bool): Should default settings for exhaustive search be used?

    """
    # split all the training data into both training and test data
    question_train, question_test, label_train, label_test = train_test_split(train_data,
                                                                              labels,
                                                                              test_size=test_size,
                                                                              random_state=random_state)
    param_svm = None
    pipeline_svm = None
    # get setup and create grid
    if use_default_settings:
        pipeline_svm = create_default_sgd_pipeline()
        param_svm = create_default_grid_parameters()
    grid_svm = GridSearchCV(pipeline_svm, param_grid=param_svm, n_jobs=-1, verbose=1)
    svm_detector = grid_svm.fit(question_train, label_train)
    dump_pickle_model(svm_detector, model_path)
    if print_results:
        print_classifier_results(svm_detector)
        if test_size > 0:
            predict_split = svm_detector.predict(question_test)
            print(confusion_matrix(label_test, predict_split))
            print(accuracy_score(label_test, predict_split))
            print(classification_report(label_test, predict_split))


# TODO: Potentially remove this in the next commit

# # retrieve data to use
# model_name = ""
# limit = DATABASE_LIMIT.get('10000')
# dataset_file = FILEPATH_TRAINING_DATA + str(limit)
# so_dataframe, pickle_model = load_classifier_model_and_dataframe(model_name, dataset_file, limit)
#
# # stem and update the data in the pandas.dataframe
# counter = 0
# corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]
# class_labels = so_dataframe.loc[:, CLASS_LABEL_KEY]
#
# for question in corpus:
#     corpus[counter] = text_processor.stem_training_data(question)
#     counter += 1
#
# # --- END: Potentially remove this in the next commit --- #


