"""
Main entry file, all user interaction is handled through this class
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mysqldatabase import MySQLDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


limit = 1000  # 10 # 100  # 1000  # 10000
class_label = MySQLDatabase.CLASS_LABEL_KEY
MySQLDatabase().set_vote_value_params()
question_text_key, training_data = MySQLDatabase().retrieve_training_data(limit)

# get and set the length of each question text
training_data['length'] = training_data[question_text_key].map(lambda text: len(text))

# question_text = training_data.get_value(index=0, col=question_text_key)
# print question_text
# print

# create a term-document matrix
count_vect = CountVectorizer(analyzer='word')
td_matrix = count_vect.fit_transform(training_data.get(question_text_key))
print td_matrix
print

#  term weighting and normalization
tfidf_transformer = TfidfTransformer().fit(td_matrix)
training_tfidf = tfidf_transformer.transform(td_matrix)

# TODO: Remove all code below; Update to match current scikit-learn and base on my dataset

# --- From tutorial: http://radimrehurek.com/data_science_python/#Step-4:-Training-a-model,-detecting-spam

# split all the training data into both training and test data (test data = 20%)
question_train, question_test, label_train, label_test = train_test_split(training_data[question_text_key],
                                                                          training_data[class_label], test_size=0.2)

pipeline_svm = Pipeline([
    ('bow', TfidfVectorizer(analyzer='word')),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# --- From tutorial: http://radimrehurek.com/data_science_python/#Step-6:-How-to-tune-parameters?

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
print svm_detector.grid_scores_

# Added for testing (fails, obviously)

# good question, ID: 927358
print svm_detector.predict(["I committed the wrong files to Git. How can I undo this commit?"])[0]
# bad question, ID: 27391628
bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
               "a header that provides the link and a manual containing instructions on how to use the library. " \
               "How can you access the private data member of the class? Note this is only specific to C++. " \
               "Normally there's no way you can access a private data member other than making " \
               "friends or writing a getter function, both of which require changing the interface of the " \
               "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
               "This is not a problem from a textbook or class assignment."
print svm_detector.predict([bad_question])[0]

print confusion_matrix(label_test, svm_detector.predict(question_test))
print classification_report(label_test, svm_detector.predict(question_test))
