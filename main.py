"""
Main entry file, all user interaction is handled through this class
"""

import pandas
from sklearn import svm
import matplotlib.pyplot as plt
from mysqldatabase import MySQLDatabase


mysql_connection = MySQLDatabase()
negative_score = mysql_connection.retrieve_posts_with_negative_votes()

# test that connection work and data gets loaded
print negative_score.get_value(index=0, col=mysql_connection.POSTS_QUESTION_TEXT)

# get and print the length of question text
negative_score['length'] = negative_score[mysql_connection.POSTS_QUESTION_TEXT].map(lambda text: len(text))
# can print via get() or via member
print negative_score.get('length'), negative_score.length

negative_score.length.plot(bins=20, kind='hist')

print negative_score.length.describe()

negative_score.hist(column='length', by='Id', bins=50)

# shows plots made by pandas
plt.show()
