"""
Main entry file, all user interaction is handled through this class
"""

from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from constants import CLASS_LABEL_KEY, QUESTION_TEXT_KEY
from python35_version.mysqldatabase import MySQLDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

mem = Memory("./mem_cache")


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
        comment = u"label: (-1: Bad question, +1: Good question); features: (term_id, length)"
        MySQLDatabase().set_vote_value_params()
        data = MySQLDatabase().retrieve_training_data(limit)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=1)
        td_matrix = vectorizer.fit_transform(data.get(QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[CLASS_LABEL_KEY], f=svm_file, comment=comment)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)

db_limit = 100  # 10 # 100  # 1000  # 10000
file_path = "./training_data/"
filename = file_path + "training_data_" + str(db_limit)

so_dataframe, (training_data, class_labels) = load_training_data(filename, False, db_limit)

# term weighting and normalization
tfidf_transformer = TfidfTransformer().fit_transform(training_data)

# exit(0)

# TODO: Remove all code below; Update to match current scikit-learn and base on my dataset

# --- From tutorial: http://radimrehurek.com/data_science_python/#Step-5:-How-to-run-experiments?

# split all the training data into both training and test data (test data = 20%)
question_train, question_test, label_train, label_test = train_test_split(so_dataframe[QUESTION_TEXT_KEY],
                                                                          class_labels,
                                                                          test_size=0.2,
                                                                          random_state=0)

# --- From tutorial: http://radimrehurek.com/data_science_python/#Step-6:-How-to-tune-parameters?

# In [42]
pipeline_svm = Pipeline([
    ('bow', TfidfVectorizer(analyzer='word', min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
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

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(question_train, label_train)  # find the best combination from param_svm
print(svm_detector.grid_scores_)

# Added for testing (fails, obviously)

# good question, ID: 927358
print(svm_detector.predict(["I committed the wrong files to Git. How can I undo this commit?"])[0])
# bad question, ID: 27391628
bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
               "a header that provides the link and a manual containing instructions on how to use the library. " \
               "How can you access the private data member of the class? Note this is only specific to C++. " \
               "Normally there's no way you can access a private data member other than making " \
               "friends or writing a getter function, both of which require changing the interface of the " \
               "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
               "This is not a problem from a textbook or class assignment."
print(svm_detector.predict([bad_question])[0])

print(confusion_matrix(label_test, svm_detector.predict(question_test)))
print(classification_report(label_test, svm_detector.predict(question_test)))
