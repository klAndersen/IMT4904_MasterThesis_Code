"""
Main entry file, all user interaction is handled through this class
"""

import pandas
from sklearn import svm
from mysqldatabase import MySQLDatabase

mysql_connection = MySQLDatabase()
negative_score = mysql_connection.retrieve_posts_with_negative_votes()
# test that connection work and data gets loaded
print negative_score.get_value(index=0, col=mysql_connection.POSTS_QUESTION_TEXT)
