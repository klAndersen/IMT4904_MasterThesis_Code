"""
Main entry file, all user interaction is handled through this class
"""

import sklearn
from sklearn import svm
import matplotlib.pyplot as plt
from mysqldatabase import MySQLDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


limit = 10

# question_text_key, negative_score = MySQLDatabase().retrieve_posts_with_negative_votes(limit)
# question_text_key1, positive_score = MySQLDatabase().retrieve_posts_with_positive_votes(limit)

question_text_key, training_data = MySQLDatabase().retrieve_training_data()

# negative_score.append(positive_score, ignore_index=True)

# print len(negative_score)
# print len(positive_score)
# # test that connection work and data gets loaded
# print negative_score.get_value(index=0, col=question_text_key)
# print positive_score.get_value(index=0, col=question_text_key)
#
# get and print the length of question text
# training_data['length'] = training_data[question_text_key].map(lambda text: len(text))
# can print via get() or via member
# print training_data.get('length'), training_data.length
#
# print training_data.label

# training_data.length.plot(bins=20, kind='hist')

# print training_data.length.describe()
#
# training_data.hist(column='length', by='Id', bins=50)
#
# shows plots made by pandas
# plt.show()
#
# print list(training_data.get(question_text_key)[training_data.length > 14400])

# tutorial: http://radimrehurek.com/data_science_python/ --- Continue from Step 2


question_text = training_data.get_value(index=0, col=question_text_key)
# "U dun say so early hor... U c already then say..." #
# ngram_range=(2, 2): binds words together, word1 + word2, word2 + word3, word3 + word4, ..., wordN-1 + wordN
# analyzer: should it be made of words or characters

vectorizer = TfidfVectorizer(analyzer='word')

# print training_data.get_value(index=0, col=question_text_key)
# print
#
# print vectorizer.build_preprocessor()(question_text)
# print
#
# tokenized_question = vectorizer.build_tokenizer()(question_text)
# print vectorizer.build_tokenizer()(question_text)
# print
#
# analysed_question = vectorizer.build_analyzer()(question_text)
# print vectorizer.build_analyzer()(question_text)
# print

bow_transformer = CountVectorizer(analyzer='word').fit(training_data.get(question_text_key))
# vectorizer.fit(training_data.get(question_text_key))

# print bow_transformer.getnnz
print "len: ", bow_transformer.vocabulary_  # fit_transform(question_text)

# get bag-of-words vector for a given question
# bow4 = bow_transformer.transform([question_text])
# print bow4
# print bow4.shape

# get feature (word) based on id/index
# print bow_transformer.get_feature_names()[307]

# sparse matrix
messages_bow = bow_transformer.transform(training_data.get(question_text_key))
print 'sparse matrix shape:', messages_bow.shape
print 'number of non-zeros:', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

#  term weighting and normalization
tfidf_transformer = TfidfTransformer().fit(messages_bow)
# tfidf4 = tfidf_transformer.transform(bow4)
# print tfidf4

# What is the IDF (inverse document frequency) of the word
print tfidf_transformer.idf_[bow_transformer.vocabulary_['what']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['how']]

# transform the entire bag-of-words corpus into TF-IDF corpus at once
messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape


