import pandas
# for Python 3.5
import mysql.connector as mysql
from mysql.connector import errorcode

import dbconfig as config
from text_processor import process_tags, remove_html_tags_from_text
from constants import QUESTION_TEXT_KEY, CLASS_LABEL_KEY, QUESTION_LENGTH_KEY, TAG_NAME_COLUMN

_author_ = "Knut Lucas Andersen"


class MySQLDatabase:
    """
    Class that handles all interactions with the MySQL database.

    The MySQL database structure can be seen in the folder ../MYSQLDB.

    The database contains all data posted on StackOverflow,
    and is based on the data set from StackExchange:

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
    (data set from March 2016), and ```negvote_Posts``` only has a file size of 1.43GB
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

    def retrieve_all_tags(self):
        """
        Retrieves all the tags in the database from 'Tags' and returns it in a list

        Returns:
            pandas.DataFrame: DataFrame containing all the tags found in the database

        """
        tag_data = None
        try:
            query = ("SELECT " + TAG_NAME_COLUMN + " FROM " + self.__TBL_TAGS + ";")
            tag_data = pandas.read_sql(query, con=self.__db)
        except mysql.Error as err:
            print("mysql.Error (retrieve_all_tags): %s", err)
        return tag_data

    def retrieve_training_data(self, limit=1000, create_feature_detectors=False, create_unprocessed=False,
                               site_tags=list, exclude_site_tags=False, exclude_assignment=False):
        """
        Retrieves Question Posts, where the questions are labeled based on if
        the question is bad (e.g. -1) or good (e.g. +1).

        The vote/score is based on the values set in ```set_vote_value_params()```
        ---> (default values are ```good: 50```, ```bad: -5```)

        Arguments:
            limit (int): Amount of rows to retrieve for each label (default=1000)
            create_feature_detectors (bool): Is this function being called to create feature detectors?
            create_unprocessed (bool): Is this function being called to create a clean, unprocessed data set?
            site_tags (list): List containing all tags found at the given site (Table: Tags)
            exclude_site_tags (bool): Should the site tags be excluded from feature detection?
            exclude_assignment (bool): Should 'assignment' words be excluded from feature detection?

        Returns:
             pandas.DataFrame: DataFrame containing the values from the database,
             plus added columns containing question length and class label
             (-1:```bad_question``` and +1:```good_question```)

        """
        training_data = pandas.DataFrame()
        # retrieve the questions with negative votes (< 0)
        class_label = -1
        score_clause = "< " + str(self.__neg_vote_value)
        negative_training_data = self.__retrieve_training_data_from_database(self.__TBL_NEGATIVE__VOTES_POSTS,
                                                                             score_clause, class_label, limit)
        # is the intention to create a fully processed data set?
        if not create_feature_detectors and not create_unprocessed:
            negative_training_data = self.__remove_html_from_text(QUESTION_TEXT_KEY,
                                                                  negative_training_data,
                                                                  site_tags, exclude_site_tags,
                                                                  exclude_assignment)
        # retrieve the questions with positive votes (> 0)
        class_label = 1
        score_clause = "> " + str(self.__pos_vote_value)
        positive_training_data = self.__retrieve_training_data_from_database(self.__TBL_POSITIVE_VOTES_POSTS,
                                                                             score_clause, class_label, limit)
        # is the intention to create a fully processed data set?
        if not create_feature_detectors and not create_unprocessed:
            positive_training_data = self.__remove_html_from_text(QUESTION_TEXT_KEY,
                                                                  positive_training_data,
                                                                  site_tags, exclude_site_tags,
                                                                  exclude_assignment)
        # add the retrieved data to the DataFrame
        training_data = training_data.append(negative_training_data, ignore_index=True)
        training_data = training_data.append(positive_training_data, ignore_index=True)
        # get and set the length of each question text
        training_data[QUESTION_LENGTH_KEY] = training_data[QUESTION_TEXT_KEY].map(lambda text: len(text))
        # close the database connection (closed here to avoid connection errors with pandas)
        self.__close_db_connection()
        return training_data

    def __retrieve_training_data_from_database(self, table_name, score_clause=str, class_label=int, limit=1000):
        """
        Retrieves all Posts data from the passed ```table_name``` and adds a class label.
        The class label indicates whether this is a good or bad question. The retrieved
        data is added to pandas.DataFrame.

        Arguments:
            table_name (str): The table to retrieve data from
            score_clause (str): WHERE clause value for score to retrieve (e.g. '> 0', '< 0', etc.)
            class_label (int): The class label for this data (e.g. -1, +1)
            limit (int): Amount of rows to retrieve (default=1000)

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
                     "Tags, "
                     "ClosedDate "
                     " FROM " + table_name + where_clause +
                     " LIMIT " + str(limit) + ";")
            posts_data = pandas.read_sql(query, con=self.__db)
            # add a column containing the class label (e.g. good/bad question, +/- 1, etc.)
            posts_data[CLASS_LABEL_KEY] = posts_data[QUESTION_TEXT_KEY].map(lambda label: class_label)
        except mysql.Error as err:
            print("mysql.Error (__retrieve_training_data_from_database): %s", err)
        return posts_data

    @staticmethod
    def __remove_html_from_text(column_name, text_data=pandas.DataFrame, site_tags=list, exclude_site_tags=False,
                                exclude_assignment=False):
        """
        Removes HTML elements (if any) from the text.

        Since all posts (questions, answers, comments, etc.) can have HTML-content,
        processing the text can become more difficult. This function loops through
        each entry in the passed DataFrame, and removes the HTML elements from the text.

        Arguments:
            column_name (str): Key to the column
            text_data (pandas.DataFrame): DataFrame with HTML text data
            site_tags (list): List containing all tags found at the given site (Table: Tags)
            exclude_site_tags (bool): Should the site tags be excluded from feature detection?
            exclude_assignment (bool): Should 'assignment' words be excluded from feature detection?

        Returns:
            pandas.DataFrame: DataFrame with updated text data

        See:
            | ```text_processor.remove_html_tags_from_text()```

        """
        invalid_question_list = list()
        # create a list of tags attached to each question, and convert the string value into arrays
        # this in turn makes them accessible by passing the index value of the given question to retrieve its tags
        tag_list = text_data["Tags"].tolist()
        tag_list = process_tags(tag_list)
        for index in range(len(text_data)):
            attached_tags = tag_list[index]
            temp_value = text_data.get_value(index=index, col=column_name)
            temp_value = temp_value.lower()
            new_value = remove_html_tags_from_text(temp_value, True, attached_tags, site_tags,
                                                   exclude_site_tags, exclude_assignment)
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
        use_default = False
        try:
            if pos_vote_value is not None:
                self.__pos_vote_value = int(pos_vote_value)
            else:
                self.__pos_vote_value = self.__DEFAULT_POSITIVE_VOTE_VALUE
            if pos_vote_value is not None:
                self.__neg_vote_value = int(neg_vote_value)
            else:
                self.__neg_vote_value = self.__DEFAULT_NEGATIVE_VOTE_VALUE
        except TypeError as err:
            print(err)
            use_default = True
        except ValueError as err:
            print(err)
            use_default = True
        if use_default:
            self.__pos_vote_value = self.__DEFAULT_POSITIVE_VOTE_VALUE
            self.__neg_vote_value = self.__DEFAULT_NEGATIVE_VOTE_VALUE

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
