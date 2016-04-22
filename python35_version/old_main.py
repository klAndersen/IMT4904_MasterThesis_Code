"""
Main entry file, all user interaction is handled through this class
"""

from pandas import DataFrame, Categorical
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from constants import CLASS_LABEL_KEY, QUESTION_TEXT_KEY
from python35_version.mysqldatabase import MySQLDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import cross_validation

from time import time, ctime
import datetime


def time_used(info, start_time):
    start_time = ctime(start_time)
    if info is None:
        print(start_time)
    else:
        print(info, start_time)
    print('\n')


def stem_training_data(question):
    porter = nltk.PorterStemmer()
    question = question.lower().split()
    m2 = map(lambda x: porter.stem(x), question)
    return ' '.join(m2)

import enchant
from nltk.corpus import words
from nltk.corpus import wordnet

# print(enchant.list_languages())

# test_word = "the"

en_US = enchant.Dict("en_US")
en_GB = enchant.Dict("en_GB")

# print(en_US.check(test_word), en_GB.check(test_word))
#
# if (wordnet.synsets(test_word)):
#     print(True)
# print(test_word in words.words())
#
# print("===IF TESTS===")
# if test_word not in words.words():
#     print("words.words()")
#
# if not wordnet.synsets(test_word):
#     print("wordnet.synsets")
#
# if not (en_US.check(test_word) or en_GB.check(test_word)):
#     print("dict_check")
#
#
# print("--COMBINED IF TEST--")
# if (test_word not in words.words()) and (not wordnet.synsets(test_word)) \
#     and not ((en_US.check(test_word) or en_GB.check(test_word))):
#     print(test_word, " is not an english word")



# exit(0)


def test_prediction_debug(classifier, vectorizer):
    """
    debugging test function

    Arguments:
        classifier (obj):
        vectorizer (TfidfVectorizer):

    """
    try:
        # good question, ID: 927358
        good_question = "I committed the wrong files to Git. How can I undo this commit?"
        good_question = stem_training_data(good_question)
        test_vector = vectorizer.transform(good_question)
        print("Predictions: ")
        print(classifier.predict(test_vector)[0])
        # bad question, ID: 27391628
        bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
                       "a header that provides the link and a manual containing instructions on how to use the library. " \
                       "How can you access the private data member of the class? Note this is only specific to C++. " \
                       "Normally there's no way you can access a private data member other than making " \
                       "friends or writing a getter function, both of which require changing the interface of the " \
                       "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
                       "This is not a problem from a textbook or class assignment."
        bad_question = stem_training_data(bad_question)
        test_vector = vectorizer.transform(bad_question)
        print(classifier.predict(test_vector)[0])

        # print(confusion_matrix(label_test, classifier.predict(question_test)))
        # print(classification_report(label_test, classifier.predict(question_test)))
    except Exception as ex:
        print(ex)

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
time_used("Program started", time_start)

db_limit = 10000  # 10 # 100  # 1000  # 10000
file_path = "./training_data/"
filename = file_path + "training_data_" + str(db_limit)

so_dataframe, (training_data, class_labels) = load_training_data(filename, False, db_limit)

# print(so_dataframe.groupby([CLASS_LABEL_KEY, QUESTION_TEXT_KEY]).describe())

# print(so_dataframe.head())
# print(so_dataframe['Id'][4])


vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
td_matrix = vectorizer.fit_transform(so_dataframe.get(QUESTION_TEXT_KEY))

print("Number of CV Features: ", td_matrix.shape)

# print("vectorizer.get_feature_names():")
# term_str = ""
# for term in vectorizer.get_feature_names():
#     term_str += term + "\n"
# print(term_str)

df = so_dataframe.loc[:, ("Score", QUESTION_TEXT_KEY, "Title", "AnswerCount", "length")]
df.loc[:, CLASS_LABEL_KEY] = Categorical(so_dataframe.loc[:, "label"])

# print(so_dataframe[CLASS_LABEL_KEY].unique())
#
# print(df.groupby(CLASS_LABEL_KEY).describe(include='all'))


# based on: http://www.irfanelahi.com/data-science-document-classification-python/#Lexical-Analysis-of-the-Text-Data

corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]

all_words = [w.split() for w in corpus]

all_flat_words = [ewords for words in all_words for ewords in words]

from nltk.corpus import stopwords

all_flat_words_ns = [w for w in all_flat_words if w not in stopwords.words("english")]
# removing all the stop words from the corpus

set_nf = set(all_flat_words_ns)


print("Number of unique vocabulary words in the " + QUESTION_TEXT_KEY + " column of the dataframe: %d" % len(set_nf))




# # TODO: TESTING
#
# from nltk.corpus import words
#
# eWords = open("non_eng_words_only.txt", "a")#Open file for writing
#
# for question in so_dataframe[QUESTION_TEXT_KEY]:  # vectorizer.get_feature_names():
#     # print(question, question.lower().split())
#     for w in question.lower().split():
#         try:
#             if w not in words.words() \
#                     and not wordnet.synsets(w) \
#                     and not (en_US.check(w) or en_GB.check(w)):#Comparing if word is non-English
#                 print('NOT: '+w)
#                 # else:#If word is an English word
#                 #     print('yes '+w)
#                 eWords.write(w)#Write to file
#         except Exception as ex:
#             print("EXCEPTION: " + w, "MSG: " + ex)
# eWords.close()#Close the file

import nltk

counter = 0
porter = nltk.PorterStemmer()
for question in so_dataframe[QUESTION_TEXT_KEY]:
    question = question.lower().split()
    m2 = map(lambda x: porter.stem(x), question)
    # Using Porter Stemmer in NLTK, stemming is performed on the str created in previous step.
    so_dataframe.loc[counter, QUESTION_TEXT_KEY] = ' '.join(m2)
    # print(question, ' '.join(m2))
    # a derived column is created and the pre-processed string is stored in that column for each row.
    counter += 1

corpus = so_dataframe.loc[:, QUESTION_TEXT_KEY]
# corpus means collection of text. For this particular data-set, I will treat the newly created column title_desc
# as my corpus and will use that to create features.
vectorizer = TfidfVectorizer(min_df=0.01, stop_words='english')
# Initializing TFIDF vectorizer to conver the raw corpus to a matrix of TFIDF features
# and also enabling the removal of stopwords.

# testing to see if results match
td_matrix = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english").fit_transform(corpus)#.todense()

tfidf_matrix = vectorizer.fit_transform(corpus).todense()
# creating TFIDF features sparse matrix by fitting it on the specified corpus.
tfidf_names = vectorizer.get_feature_names()
# grabbing the name of the features.

# TODO: add note about how much feature reduction the filters added

# comparing tutorial vs mine
print("Number of CV Features: ", td_matrix.shape)
print("Number of TFIDF Features: %d" % len(tfidf_names), " ", tfidf_matrix.shape)  # same info can be gathered by using tfidf_matrix.shape

# for name in tfidf_names:
#     print(name)

# exit(0)

# TODO: TESTING

from nltk.corpus import words

eWords = open("non_eng_words_only.txt", "a")#Open file for writing

#### MUST BE RUN BEFORE PORTER_STEMMING; see http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
# for w in vectorizer.get_feature_names():
#     try:
#         if w not in words.words() \
#                 and not wordnet.synsets(w) \
#                 and not (en_US.check(w) or en_GB.check(w)):#Comparing if word is non-English
#             print('NOT: '+w)
#             # else:#If word is an English word
#             #     print('yes '+w)
#             eWords.writelines(w)#Write to file
#     except Exception as ex:
#         print("EXCEPTION: " + w, "MSG: " + ex)
# eWords.close()#Close the file

time_used("Splitting test set", time())

training_time_container = {'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
prediction_time_container = {'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}

accuracy_container = {'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}

# split all the training data into both training and test data (test data = 20%)
question_train, question_test, label_train, label_test = train_test_split(td_matrix,
                                                                          class_labels,
                                                                          test_size=0.2,
                                                                          random_state=0)

# for label in label_test:
#     print(label)

time_used("Starting training", time())

#analyzing the shape of the training and test data-set:
print('Shape of Training Data: '+str(question_train.shape))
print('Shape of Test Data: '+str(question_test.shape))

from sklearn.naive_bayes import BernoulliNB
#loading Gaussian Naive Bayes from the sklearn library:
bnb_classifier=BernoulliNB()
#initializing the object
t0=time()
bnb_classifier=bnb_classifier.fit(question_train,label_train)
training_time_container['b_naive_bayes']=time()-t0
#fitting the classifier or training the classifier on the training data

#after the model has been trained, we proceed to test its performance on the test data:
t0=time()
bnb_predictions=bnb_classifier.predict(question_test)
prediction_time_container['b_naive_bayes']=time()-t0

#the trained classifier has been used to make predictions on the test data-set. To evaluate the performance of the model,
#there are a number of metrics that can be used as follows:
nb_ascore=accuracy_score(label_test, bnb_predictions)
accuracy_container['b_naive_bayes']=nb_ascore

print("Bernoulli Naive Bayes Accuracy Score: %f"%accuracy_container['b_naive_bayes'])
print("Training Time: %f"%training_time_container['b_naive_bayes'])
print("Prediction Time: %f"%prediction_time_container['b_naive_bayes'])

#it shows that the accuracy score of our model is 0.954 or 95.4%.
#Confusion Matrix is also another way to evaluate the prediction output of a classifier and also to determine the false positive
#and false negative, sensitivity, specificity, precision and recall metrics:
print("Confusion Matrix of Bernoulli Naive Bayes Classifier output: ")
print(confusion_matrix(label_test,bnb_predictions))
#the values on the diagonal show correct predictions where as off-diagonal represent the records that have been misclassified.

print("Classification Metrics: ")
print(classification_report(label_test,bnb_predictions))
#accuracy score can be misleading when there is class imbalance problem in the data-set. In our case, the problem wasn't that
#significant. F1-Score is a better measure of a classifier performance. The greater the F1-Score, the better.

from sklearn.naive_bayes import MultinomialNB
mn_bayes=MultinomialNB()
t0=time()
mn_bayes_fit=mn_bayes.fit(question_train,label_train)
training_time_container['mn_naive_bayes']=time()-t0
t0=time()
prediction_mn=mn_bayes_fit.predict(question_test)
prediction_time_container['mn_naive_bayes']=time()-t0
mn_ascore=accuracy_score(label_test, prediction_mn)
accuracy_container['mn_naive_bayes']=mn_ascore

#if we see the accuracy score of Multinomial Naive Bayes classifier, we come to see that it turns out be around 0.934 or 93.4%
print("Accuracy Score of Multi-Nomial Naive Bayes: %f" %(mn_ascore))
#and its training and prediction time are:
print("Training Time: %fs"%training_time_container['mn_naive_bayes'])
print("Prediction Time: %fs"%prediction_time_container['mn_naive_bayes'])

from sklearn.ensemble import RandomForestClassifier

rf_classifier=RandomForestClassifier(n_estimators=50)
t0=time()
rf_classifier=rf_classifier.fit(question_train,label_train)

training_time_container['random_forest']=time()-t0
print("Training Time: %fs"%training_time_container['random_forest'])

t0=time()
rf_predictions=rf_classifier.predict(question_test)
prediction_time_container['random_forest']=time()-t0
print("Prediction Time: %fs"%prediction_time_container['random_forest'])

accuracy_container['random_forest']=accuracy_score(label_test, rf_predictions)
print ("Accuracy Score of Random Forests Classifier: ")
print(accuracy_container['random_forest'])
print(confusion_matrix(label_test,rf_predictions))

#I've used hinge loss which gives linear Support Vector Machine. Also set the learning rate to 0.0001 (also the default value)
#which is a constant that's gets multiplied with the regularization term. For penalty, I've used L2 which is the standard
#regularizer for linear SVMs:

from sklearn import linear_model

svm_classifier=linear_model.SGDClassifier(loss='hinge',alpha=0.0001)

t0=time()
svm_classifier=svm_classifier.fit(question_train, label_train)
training_time_container['linear_svm']=time()-t0
print("Training Time: %fs"%training_time_container['linear_svm'])

t0=time()
svm_predictions=svm_classifier.predict(question_test)
prediction_time_container['linear_svm']=time()-t0
print("Prediction Time: %fs"%prediction_time_container['linear_svm'])

accuracy_container['linear_svm']=accuracy_score(label_test, svm_predictions)
print ("Accuracy Score of Linear SVM Classifier: %f"%accuracy_container['linear_svm'])
print(confusion_matrix(label_test,svm_predictions))

#if we train the SGD Classifier with elastic net penalty, it  brings more sparsity to the model not possible with the L2:
svm_classifier_enet=linear_model.SGDClassifier(loss='hinge',alpha=0.0001,penalty='elasticnet')
svm_classifier_enet=svm_classifier_enet.fit(question_train, label_train)

svm_enet_predictions=svm_classifier_enet.predict(question_test)

print ("Accuracy Score of Linear SVM Classifier: %f"%accuracy_score(label_test,svm_enet_predictions))
# we saw marginal improvement in the overall accuracy score.

from sklearn.svm import SVC

nl_svm_classifier=SVC(C=1000000.0, gamma='auto', kernel='rbf')

t0=time()
nl_svm_classifier=nl_svm_classifier.fit(question_train,label_train)
training_time_container['non_linear_svm']=time()-t0

t0=time()
nl_svm_predictions=nl_svm_classifier.predict(question_test)
prediction_time_container['non_linear_svm']=time()-t0

accuracy_container['non_linear_svm']=accuracy_score(label_test,nl_svm_predictions)

print("Accuracy score of Non-Linear SVM: %f"%accuracy_container['linear_svm'])


# test predictions

# good_question = "I committed the wrong files to Git. How can I undo this commit?"
# good_question = stem_training_data(good_question)
# print(good_question[0], good_question, so_dataframe[QUESTION_TEXT_KEY][0], vectorizer.get_feature_names()[0])
# test_vector = vectorizer.transform(good_question)

# print("Predictions: ")
# print(bnb_classifier.predict(test_vector)[0])
# bad question, ID: 27391628
# bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
#                "a header that provides the link and a manual containing instructions on how to use the library. " \
#                "How can you access the private data member of the class? Note this is only specific to C++. " \
#                "Normally there's no way you can access a private data member other than making " \
#                "friends or writing a getter function, both of which require changing the interface of the " \
#                "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
#                "This is not a problem from a textbook or class assignment."
# bad_question = stem_training_data(bad_question)
# test_vector = vectorizer.transform(bad_question)
# print(bnb_classifier.predict(test_vector))

test_prediction_debug(bnb_classifier, vectorizer)
test_prediction_debug(mn_bayes, vectorizer)
test_prediction_debug(rf_classifier, vectorizer)
test_prediction_debug(svm_classifier, vectorizer)
test_prediction_debug(nl_svm_classifier, vectorizer)

import matplotlib.pyplot as plt

with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(12,7))
    plt.bar(range(5),training_time_container.values(),tick_label=training_time_container.keys(),align='center')
    plt.ylabel("Training time in seconds")
    plt.ylim(0,8)
    plt.grid(True)
    plt.title("Comparison of Training Time of different classifiers")

with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(12,7))
    plt.bar(range(5),prediction_time_container.values(),tick_label=prediction_time_container.keys(),align='center',color='orange')

    plt.ylabel("Prediction time in seconds")
    plt.grid(True)
    plt.ylim(0,1)
    plt.title("Comparison of Prediction Time of different classifiers")


with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(12,7))
    plt.bar(range(5),accuracy_container.values(),tick_label=accuracy_container.keys(),align='center',color='g')

    plt.ylabel("Accuracy Scores")
    plt.grid(True)
    plt.title("Comparison of Accuracy Scores of different classifiers")
    plt.ylim(0.85,1.0)


time_used("Analysing results with SelectKBest and chi2", time_start)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2, k='all').fit_transform(td_matrix, class_labels)

#splititng the data into training and test data-set again:
cvariables_train, cvariables_test, clabels_train, clabels_test = train_test_split(X_new, class_labels, test_size=.3)

#now use these features to train the linear SVM Classifier and see what results do you get:
svm_classifier2=linear_model.SGDClassifier(alpha=0.0001,penalty='elasticnet',n_iter=50)
svm_classifier_f2=svm_classifier2.fit(cvariables_train, clabels_train)
predictions_svm2=svm_classifier_f2.predict(cvariables_test)

confusion_matrix(clabels_test,predictions_svm2)

#and if we analyze the accuracy score, we can clearly see that its marginally better than the previous accuracy score that we got
#with Linear SVM trained on training data with 13k+  features. So its a remarkable improvement that using less than half features
#we are able to achieve equivalently good accuracy score.
accuracy_score(clabels_test,predictions_svm2)

mn_bayes=MultinomialNB()
cv_scores = cross_validation.cross_val_score(mn_bayes, question_train, label_train, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
#so the results that we've obtained with cross validation techniques are also almost equivalent to the ones we obtained when
#we trained the Multinomial naive bayes classifier before. This further validates our results and accuracy scores that we've
#got in the analysis.
#F1-score can also be calculated as follows:
cv_scores_f1=cross_validation.cross_val_score(mn_bayes,question_train,label_train,cv=5,scoring='f1_weighted')

print("F1-Scores of Naive Bayes Classifier on Cross Validation Data: %f"%cv_scores_f1.mean())
#which is also approximately equal to the accuracy score that we obtained.

print(accuracy_container['mn_naive_bayes'])

time_start = time()
time_used("Starting SVM & GridSearch", time_start)

# '''
# --- From tutorial: http://radimrehurek.com/data_science_python/#Step-6:-How-to-tune-parameters?

# split all the training data into both training and test data (test data = 20%)
question_train, question_test, label_train, label_test = train_test_split(so_dataframe[QUESTION_TEXT_KEY],
                                                                          class_labels,
                                                                          test_size=0.2,
                                                                          random_state=0)

# In [42]
pipeline_svm = Pipeline([
    ('bow', TfidfVectorizer(analyzer='word', min_df=0.01, stop_words='english')),
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
    # {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['sigmoid']},
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
# '''

# good question, ID: 927358
good_question = "I committed the wrong files to Git. How can I undo this commit?"
good_question = stem_training_data(good_question)
print("Predictions: ")
print(svm_detector.predict([good_question])[0])
# bad question, ID: 27391628
bad_question = "You like C++ a lot. Now you have a compiled binary file of a library, " \
               "a header that provides the link and a manual containing instructions on how to use the library. " \
               "How can you access the private data member of the class? Note this is only specific to C++. " \
               "Normally there's no way you can access a private data member other than making " \
               "friends or writing a getter function, both of which require changing the interface of the " \
               "said class. C++ is a bit different in that you can think of it as a wrapper of C. " \
               "This is not a problem from a textbook or class assignment."
bad_question = stem_training_data(bad_question)
print(svm_detector.predict([bad_question])[0])

# print(confusion_matrix(label_test, classifier.predict(question_test)))
# print(classification_report(label_test, classifier.predict(question_test)))

# TODO: Note, it now takes an hour (+/-) for svm training to complete

time_start = time()
time_used("Finished all tasks", time_start)
