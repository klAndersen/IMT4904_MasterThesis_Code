import pandas
# for Python 3.5
import mysql.connector as mysql
from mysql.connector import errorcode

import text_processor
import dbconfig as config
from constants import QUESTION_TEXT_KEY, CLASS_LABEL_KEY, QUESTION_LENGTH_KEY

_author_ = "Knut Lucas Andersen"


class MySQLDatabase:
    """
    Class that handles all interactions with the MySQL database.

    The MySQL database structure can be seen in the folder ../MYSQLDB.

    The database contains all data posted on StackOverflow,
    and is based on the dataset from StackExchange:

    https://archive.org/details/stackexchange
    """

    # "Constant" values: Primary keys
    __PK_IS_ID = "Id"
    __TBL_BADGES_ID = "UserId"
    # "Constant" values: Table names
    __TBL_BADGES = "Badges"
    __TBL_COMMENTS = "Comments"
    __TBL_POSTHISTORY = "PostHistory"
    __TBL_POSTS = "Posts"
    __TBL_TAGS = "Tags"
    __TBL_VOTES = "Votes"
    __TBL_POSITIVE_VOTES_POSTS = "posvote_Posts"
    __TBL_NEGATIVE__VOTES_POSTS = "negvote_Posts"
    '''
    To reduce data amount when retrieving training data, a table was
    created which contains only Questions with votes/score < 0.

    Size-wise this was beneficial, since the Posts has a file size of 44.1GB
    (dataset from March 2016), and ```negvote_Posts``` only has a file size of 1.43GB
    '''
    # Values for votes in WHERE clause
    __DEFAULT_POSITIVE_VOTE_VALUE = int(50)
    '''
    Default value to use in WHERE clause for good questions

    (Value = 50)
    '''
    __DEFAULT_NEGATIVE_VOTE_VALUE = int(-5)
    '''
    Default value to use in WHERE clause for bad questions

    (Value = -5)
    '''

    def __init__(self):
        """
        Constructor connecting to the MySQL database.
        """
        try:
            # retrieve connection parameters from config file
            self.__mysql_parameters = config.mysql_parameters
            # attempt to connect to the database
            self.__db = mysql.connect(**self.__mysql_parameters)
            self.__pos_vote_value = self.__DEFAULT_POSITIVE_VOTE_VALUE
            self.__neg_vote_value = self.__DEFAULT_NEGATIVE_VOTE_VALUE
        except mysql.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    def retrieve_training_data(self, limit=1000, remove_html=True):
        """
        Retrieves Question Posts, where the questions are labeled based on if
        the question is bad (e.g. -1) or good (e.g. +1).

        The vote/score is based on the values set in ```set_vote_value_params()```
        ---> (default values are ```good: 50```, ```bad: -5```)

        Arguments:
            limit (int): Amount of rows to retrieve for each label (default=1000)
            remove_html (bool): Should HTML be removed from the question text?

        Returns:
             pandas.DataFrame: DataFrame containing the values from the database,
             plus added columns containing question length and class label
             (-1:```bad_question``` and +1:```good_question```)

        """
        training_data = pandas.DataFrame()
        # retrieve the questions with negative votes (< 0)
        class_label = -1
        score_clause = "< " + str(self.__neg_vote_value)
        negative_training_data = self.retrieve_question_posts(self.__TBL_NEGATIVE__VOTES_POSTS,
                                                              class_label, score_clause, limit, remove_html)
        # retrieve the questions with positive votes (> 0)
        class_label = 1
        score_clause = "> " + str(self.__pos_vote_value)
        positive_training_data = self.retrieve_question_posts(self.__TBL_POSITIVE_VOTES_POSTS,
                                                              class_label, score_clause, limit, remove_html)
        # add the retrieved data to the DataFrame
        training_data = training_data.append(negative_training_data, ignore_index=True)
        training_data = training_data.append(positive_training_data, ignore_index=True)
        # close the database connection (closed here to avoid connection errors with pandas)
        self.__close_db_connection()
        return training_data

    def retrieve_question_posts(self, table_name, class_label, score_clause=str, limit=1000, remove_html=True):
        """
        Retrieves all Posts data from the passed ```table_name```, removes the HTML from the
        question body and adds a class label. The class label indicates whether this is a good
        or bad question. The retrieved data is added to pandas.DataFrame.

        Arguments:
            table_name (str): The table to retrieve data from
            class_label (int): The class label for this data (e.g. -1, +1)
            score_clause (str): WHERE clause value for score to retrieve (e.g. '> 0', '< 0', etc.)
            limit (int): Amount of rows to retrieve (default=1000)
            remove_html (bool): Should HTML be removed from the question text?

        Returns:
            pandas.DataFrame: DataFrame containing the retrieved data || None

        """
        posts_data = None
        try:
            where_clause = ''
            if score_clause:
                where_clause = " WHERE Score " + score_clause
            query = ("SELECT Id, "
                     "Score, "
                     "ViewCount, "
                     "Body, "
                     "Title, "
                     "AnswerCount, "
                     "CommentCount, "
                     "AcceptedAnswerId, "
                     "OwnerUserId, "
                     "CreationDate, "
                     "ClosedDate "
                     " FROM " + table_name + where_clause +
                     " LIMIT " + str(limit) + ";")
            posts_data = pandas.read_sql(query, con=self.__db)
            if remove_html:
                posts_data = self.__remove_html_from_text(QUESTION_TEXT_KEY, posts_data)
            # add a column containing the class label (e.g. good/bad question, +/- 1, etc.)
            posts_data[CLASS_LABEL_KEY] = posts_data[QUESTION_TEXT_KEY].map(lambda label: class_label)
            # get and set the length of each question text
            posts_data[QUESTION_LENGTH_KEY] = posts_data[QUESTION_TEXT_KEY].map(lambda text: len(text))
        except mysql.Error as err:
            print("mysql.Error (retrieve_question_posts): %s", err)
        return posts_data

    @staticmethod
    def __remove_html_from_text(column_name, text_data=pandas.DataFrame):
        """
        Removes HTML elements (if any) from the text.

        Since all posts (questions, answers, comments, etc.) can have HTML-content,
        processing the text can become more difficult. This function loops through
        each entry in the passed DataFrame, and removes the HTML elements from the text.

        Arguments:
            column_name (str): Key to the column
            text_data (pandas.DataFrame): DataFrame with HTML text data

        Returns:
            pandas.DataFrame: DataFrame with updated text data

        See:
            | ```htmlstripper.strip_tags()```

        """
        invalid_question_list = list()
        for index in range(len(text_data)):
            temp_value = text_data.get_value(index=index, col=column_name)
            new_value = text_processor.strip_tags(temp_value)
            if new_value is None:
                new_value = temp_value
                invalid_question_list.append(index)
            text_data.set_value(index=index, col=column_name, value=new_value)
        # in case some questions couldn't be properly HTML cleansed, remove them from the data set
        counter = len(invalid_question_list) - 1
        while counter > -1:
            removal_index = invalid_question_list[counter]
            text_data.drop(removal_index, inplace=True)
            counter -= 1
        return text_data

    def select_all_records_from_table(self, table_name, limit=1000):
        """
        Retrieves n (where n=limit) records from the selected table

        Arguments:
            table_name (str): Name of table to retrieve data from
            limit (int): Restriction for how many records to retrieve

        Returns:
            dict: Dictionary containing the result of the query || None

        """
        result_set = None
        try:
            cursor = self.__db.cursor()
            query = "SELECT * FROM " + table_name + " LIMIT " + str(limit) + ";"
            # run query and get the results
            cursor.execute(query)
            result_set = cursor.fetchall()
        except mysql.Error as err:
            # print cursor._last_executed
            print("mysql.Error (Select all): %s", err)
        finally:
            self.__close_db_connection()
        return result_set

    def set_vote_value_params(self, pos_vote_value=int, neg_vote_value=int):
        """
        Sets the start value for the question votes to retrieve for the training

        Arguments:
            pos_vote_value (int): Start value for good questions (>= 0).
            neg_vote_value (int): Start value for bad questions (<= 0).

        """
        self.__pos_vote_value = pos_vote_value
        self.__neg_vote_value = neg_vote_value

    def __close_db_connection(self):
        """
        Closes the database connection
        """
        try:
            self.__db.close()
        except mysql.ProgrammingError as err:
            # if the error is that the connection is already closed, ignore it
            if str(err) == "closing a closed connection":
                pass
            else:
                print("mysql.ProgrammingError: %s", err)
        except mysql.Error as err:
            print("mysql.Error (Close Connection): %s", err)
