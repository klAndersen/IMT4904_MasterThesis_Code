"""
Main entry file, all user interaction is handled through this class
"""

import nltk
import pickle
from time import time, ctime
from pandas import DataFrame

from constants import CLASS_LABEL_KEY, QUESTION_TEXT_KEY
from python35_version.mysqldatabase import MySQLDatabase

from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

mem = Memory("./mem_cache")


def current_time(time_now, info):
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
current_time("Program started", time_start)

db_limit = 10000  # 10 # 100  # 1000  # 10000
file_path = "./training_data/"
filename = file_path + "training_data_" + str(db_limit)

so_dataframe, (training_data, class_labels) = load_training_data(filename)

# if __name__ == "__main__":
