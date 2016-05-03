"""
Main entry file, all user interaction is handled through this class
"""

# python imports
import nltk
import pickle
from time import time, ctime
from pandas import DataFrame
from os import listdir
from os.path import isfile, join
# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# project imports
from constants import *
from mysqldatabase import MySQLDatabase


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


def get_training_model(path=str, model_name=str, suffix=".pkl"):
    """
    Returns the model with the given name at the given path if it exists

    Arguments:
        path (str): The directory path containing the training model(s)
        model_name (str): The filename of the model to retrieve
        suffix (str): The models file type (expects pickle type)

    Returns:
        model: The loaded pickle model || None if error occurred

    """
    try:
        # retrieve only the files with given suffix
        selected_files_only = [
            file for file in listdir(path)
            if (isfile(join(path, file)) and file.endswith(suffix))
            ]
        # was there any files in the given path, and does the model exists?
        if len(selected_files_only) > 0:
            file = model_name + suffix
            files_found = '\n'.join(map(lambda f: f, selected_files_only))
            if file in selected_files_only:
                file = path + file
                model = load_pickle_model(file)
                if model is not None:
                    feedback = "Model '" + model_name + suffix + "' loaded."
                    print(feedback)
                    return model
                else:
                    feedback = "Could not load the model '" + model_name + suffix + "'."
            else:
                feedback = "No classifier model named '" + model_name + suffix + "' found in '" + path + "."
            if feedback is not None:
                feedback += "\nThe following models were found: \n" + files_found
                print(feedback)
        else:
            feedback = "No classifier models found in '" + path + "."
            print(feedback)
    except Exception as ex:
        print("Failed at loading training model: ", ex)
    return None


def print_startup_menu():
    """
    Prints the startup menu displayed to the user on first run
    """
    menu = "Menu:"
    for key in sorted(USER_MENU_OPTIONS):
        temp_dict = USER_MENU_OPTIONS.get(key)
        menu += key + ": " + temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY) + "\n"
    print(menu)


def load_default_model():
    """
    Loads an existing model that has been previously created

    Returns:
        model: The loaded pickle model || None if error occurred
    """
    limit = DATABASE_LIMIT.get('10000')
    mod_split_data = "svm_detector_split_" + str(limit)
    print(mod_split_data)
    return get_training_model(FILEPATH_MODELS, mod_split_data)


def end_program():
    print("Exiting program...")
    exit(0)


def create_new_training_model():
    # TODO: Create a new model based on passed parameters
    # path, name, limit, classifier_data
    pass


def predict_question_quality(model, question):
    # TODO: Make a prediction based on loaded model and entered question
    # model, question (needs to be reconstructed to a string)
    # question would then need to be processed and then controlled against model
    pass


def handle_user_input(model, u_input=str):
    """
    Takes the user input and checks what operation to execute

    Arguments:
        model: Classifier model loaded from pickle || None
        u_input (str):

    Returns:
        model: Classifier model loaded from pickle || None

    """
    args = None
    if len(u_input) > 1:
        command, *args = u_input.split()
    else:
        command = u_input
    if command == USER_MENU_OPTION_HELP_KEY:
        print_startup_menu()
    elif command == USER_MENU_OPTION_NEW_PREDICTION:
        if model is None:
            print("No model is loaded. Load default model by entering 'd' or 'l path filename suffix'.")
        else:
            predict_question_quality(model, question)
    elif command == USER_MENU_OPTION_LOAD_DEFAULT_KEY:
        return load_default_model()
    elif command == USER_MENU_OPTION_LOAD_USER_MODEL_KEY:
        print(len(args))
    elif command == USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY:
        return create_new_training_model()
    elif command != USER_MENU_OPTION_EXIT_KEY:
        print("Invalid command: ", command)
    return model


if __name__ == "__main__":
    user_input = ""
    classifier_model = None
    print_startup_menu()
    while user_input != USER_MENU_OPTION_EXIT_KEY:
        try:
            user_input = input("Enter option followed by arguments (if any) Enter h to show options: ")
            classifier_model = handle_user_input(user_input)
        except ValueError as ex:
            print("ValueError: ", ex)

    if user_input == USER_MENU_OPTION_EXIT_KEY:
        end_program()


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
    stemming_data = map(lambda x: porter.stem(x), stemming_data)
    return ' '.join(stemming_data)


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


time_start = time()
current_time(time_start, "Program started")
# retrieve data to use
db_limit = DATABASE_LIMIT.get('10000')
filename = FILEPATH_TRAINING_DATA + str(db_limit)
so_dataframe, (training_data, class_labels) = load_training_data(filename, False, db_limit, True)

# stem and update the data in the pandas.dataframe
counter = 0
for question in so_dataframe[QUESTION_TEXT_KEY]:
    so_dataframe.loc[counter, QUESTION_TEXT_KEY] = stem_training_data(question)
    counter += 1

corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]

pickle_exists = True
create_model_from_all_data = False
# set paths for model retrieval
mod_all_data_path = FILEPATH_MODELS + "svm_detector_all_" + str(db_limit) + ".pkl"
mod_split_data_path = FILEPATH_MODELS + "svm_detector_split_" + str(db_limit) + ".pkl"
# split all the training data into both training and test data (test data = 20%)
question_train, question_test, label_train, label_test = train_test_split(corpus,
                                                                          class_labels,
                                                                          test_size=0.2,
                                                                          random_state=0)

if __name__ == "__main__":
    if pickle_exists:
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
            dump_pickle_model(svm_detector_all, mod_all_data_path)
        else:
            svm_detector_split = grid_svm.fit(question_train, label_train)
            dump_pickle_model(svm_detector_split, mod_split_data_path)
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
        predict_split = svm_detector_split.predict(question_test)
        print(confusion_matrix(label_test, predict_split))
        print(accuracy_score(label_test, predict_split))
        print(classification_report(label_test, predict_split))
