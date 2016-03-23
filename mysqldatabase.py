
import pandas
import MySQLdb

import htmlstripper
import dbconfig as config

_author_ = "Knut Lucas Andersen"


class MySQLDatabase:
    """
    Class for handling MySQL database operations.
    """

    __db = None  # the database connection
    __mysql_parameters = None  # parameters for database connection
    # Constant values: Error values
    PRIMARY_KEY_NOT_FOUND = -1
    # "Constant" values: Table names
    __TBL_BADGES = "Badges"
    __TBL_COMMENTS = "Comments"
    __TBL_POSITIVE_SCORED_POSTS = "posvote_Posts"
    __TBL_NEGATIVE_SCORED_POSTS = "negvote_Posts"
    '''
    To reduce data amount when retrieving training data, a table was
    created which contains only Questions with votes/score < 0.
    Size-wise this was beneficial, since the Posts has a file size of 30.6GB,
    and this only has a file size of 0.7GB
    '''
    __TBL_POSTHISTORY = "PostHistory"
    __TBL_POSTS = "Posts"
    __TBL_TAGS = "Tags"
    __TBL_VOTES = "Votes"
    # "Constant" values: Primary keys
    __PK_IS_ID = "Id"
    __TBL_BADGES_ID = "UserId"
    # "Constant" values: Column names
    POSTS_VOTES_KEY = "Score"
    '''
    Column identifier/key: Score
    Amount of votes/scores on a given question
    '''
    POSTS_TITLE_KEY = "Title"
    '''
    Column identifier/key: Title
    The title of the question as seen on StackOverflow
    '''
    POSTS_QUESTION_TEXT_KEY = "Body"
    '''
    Column identifier/key: Body
    The question text
    '''
    CLASS_LABEL_KEY = "label"
    '''
    Column identifier/key: label
    Label for the retrieved training data.
    '''

    def __init__(self):
        """
        Constructor connecting to the MySQL database.
        """
        try:
            # retrieve connection parameters from config file
            self.__mysql_parameters = config.mysql_parameters
            # attempt to connect to the database
            self.__db = MySQLdb.connect(
                self.__mysql_parameters['host'],
                self.__mysql_parameters['user'],
                self.__mysql_parameters['passwd'],
                self.__mysql_parameters['db']
            )
            self.__db.autocommit(True)
        except MySQLdb.Error as err:
            print("Error during connection: %s", err)

    def retrieve_question_posts(self, table_name, class_label, limit=1000, remove_html=True):
        """

        Arguments:
            table_name (str): The table to retrieve data from
            class_label (str): The class label for this data
            limit (int): Amount of rows to retrieve (default=1000)
            remove_html (bool): Should HTML be removed from the question text

        Returns:
            pandas.DataFrame:

        """
        posts_data = None
        try:
            query = "SELECT * FROM " + table_name \
                    + " LIMIT " + str(limit) + ";"
            posts_data = pandas.read_sql(query, con=self.__db)
            if remove_html:
                posts_data = self.__remove_html_from_text(self.POSTS_QUESTION_TEXT_KEY, posts_data)
            # add a column containing the class label (e.g. good/bad question)
            posts_data[self.CLASS_LABEL_KEY] = posts_data[self.POSTS_QUESTION_TEXT_KEY].map(lambda text: class_label)
        except MySQLdb.Error as err:
            print("MySQLdb.Error (retrieve_question_posts): %s", err)
        return posts_data

    def retrieve_training_data(self, limit=1000, remove_html=True):
        """
        Retrieves all Posts (questions only) where the questions are labeled
        based on if the question is bad (score < 0) or good (score > 0)

        Arguments:
            limit (int): Amount of rows to retrieve for each label (default=1000)
            remove_html (bool): Should HTML be removed from the question text

        Returns:
             tuple(str, pandas.DataFrame): The str value is key to the column containing the
                                       question text, and the DataFrame contains the labeled
                                       training data (```bad_question``` and ```good_question```)

        """
        training_data = pandas.DataFrame()
        # retrieve the questions with negative votes (< 0)
        class_label = "bad_question"
        negative_training_data = self.retrieve_question_posts(self.__TBL_NEGATIVE_SCORED_POSTS,
                                                              class_label, limit, remove_html)
        # retrieve questions with votes > 0
        class_label = "good_question"
        positive_training_data = self.retrieve_question_posts(self.__TBL_POSITIVE_SCORED_POSTS,
                                                              class_label, limit, remove_html)
        # add the retrieved data to the DataFrame
        training_data = training_data.append(negative_training_data, ignore_index=True)
        training_data = training_data.append(positive_training_data, ignore_index=True)
        # close the database connection (closed here to avoid connection errors with pandas)
        self.__close_db_connection()
        return self.POSTS_QUESTION_TEXT_KEY, training_data
    #
    # def retrieve_posts_with_positive_votes(self, limit=1000, remove_html=True):
    #     """
    #     Retrieves all Posts (questions only) where the voting score is positive (> 0)
    #
    #     Arguments:
    #         limit: Amount of rows to retrieve (default=1000)
    #         remove_html: Should HTML be removed from the question text
    #
    #     Returns:
    #          tuple(str, pandas.DataFrame): The str value is key to the column containing the
    #                                    question text, and the DataFrame contains question posts
    #                                    with a voting score > 0
    #
    #     """
    #     posts_data = None
    #     question_text_key = None
    #     try:
    #         query = "SELECT * FROM " + self.__TBL_POSITIVE_SCORED_POSTS \
    #                 + " LIMIT " + str(limit) + ";"
    #         posts_data = pandas.read_sql(query, con=self.__db)
    #         question_text_key = self.__POSTS_QUESTION_TEXT
    #         if remove_html:
    #             posts_data = self.__remove_html_from_text(question_text_key, posts_data)
    #         # add a column containing the class label (e.g. good/bad question)
    #         posts_data[self.CLASS_LABEL] = posts_data[question_text_key].map(lambda text: 'good')
    #     except MySQLdb.Error as err:
    #         print("MySQLdb.Error (retrieve_posts_with_positive_votes): %s", err)
    #     finally:
    #         self.__close_db_connection()
    #     return question_text_key, posts_data
    #
    # def retrieve_posts_with_negative_votes(self, limit=1000, remove_html=True):
    #     """
    #     Retrieves all Posts (questions only) where the voting score is negative (< 0)
    #
    #     Arguments:
    #         limit: Amount of rows to retrieve (default=1000)
    #         remove_html: Should HTML be removed from the question text
    #
    #     Returns:
    #          tuple(str, pandas.DataFrame): The str value is key to the column containing the
    #                                    question text, and the DataFrame contains question posts
    #                                    with a voting score < 0
    #
    #     """
    #     posts_data = None
    #     question_text_key = None
    #     try:
    #         query = "SELECT * FROM " + self.__TBL_NEGATIVE_SCORED_POSTS \
    #                 + " LIMIT " + str(limit) + ";"
    #         posts_data = pandas.read_sql(query, con=self.__db)
    #         question_text_key = self.__POSTS_QUESTION_TEXT
    #         if remove_html:
    #             posts_data = self.__remove_html_from_text(question_text_key, posts_data)
    #         # add a column containing the class label (e.g. good/bad question)
    #         posts_data[self.CLASS_LABEL] = posts_data[question_text_key].map(lambda text: 'bad')
    #     except MySQLdb.Error as err:
    #         print("MySQLdb.Error (retrieve_posts_with_negative_votes): %s", err)
    #     finally:
    #         self.__close_db_connection()
    #     return question_text_key, posts_data

    @staticmethod
    def __remove_html_from_text(column_name, text_data=pandas.DataFrame):
        """
        Removes HTML elements (if any) from the text

        Since all posts (questions, answers, comments, etc) can have HTML-content,
        processing the text can become more difficult. This function loops through
        each entry in the passed DataFrame, and removes the HTML elements from the
        text.

        Arguments:
            column_name (str): Key to the column
            text_data (pandas.DataFrame): DataFrame with HTML text data

        Returns:
            pandas.DataFrame: DataFrame with updated text data

        See:
            | ```htmlstripper.strip_tags()```

        """
        for index in range(len(text_data)):
            temp_value = text_data.get_value(index=index, col=column_name)
            new_value = htmlstripper.strip_tags(temp_value.decode("utf-8"))
            text_data.set_value(index=index, col=column_name, value=new_value)
        return text_data

    def select_all_records_from_tables(self, table_list, limit=1000):
        """
        Retrieves all records from the selected tables.

        This function takes a list as input, where you can add one or more table names
        to retrieve all the data from. The passed list is converted to a tuple for
        string interpolation. Therefore, user data should not be passed to this function,
        since table names cannot be parametrized (it is neither the intention that this
        function should be called/used through user input).

        Arguments:
            table_list (list): List containing the names of tables to retrieve data from
            limit (int): Restriction for how many records to retrieve

        See:
            | ```MySQLCursor.fetchall()```
            | ```MySQLdb.cursors.DictCursor```

        Returns:
            dict: Dictionary containing the result of the query || None

        """
        result_set = None
        try:
            cursor = self.__get_db_cursor()
            query = "SELECT * FROM %s"
            no_of_entries = len(table_list) - 1
            # if there are more then one table in the list...
            if no_of_entries > 0:
                for counter in range(0, no_of_entries):
                    query += ", %s "
            # add semi-colon at the end of the query
            query += " LIMIT " + str(limit) + ";"
            # run query and get the results
            cursor.execute(query % tuple(table_list))
            result_set = cursor.fetchall()
        except MySQLdb.Error as err:
            # print cursor._last_executed
            print("MySQLdb.Error (Select all): %s", err)
        finally:
            self.__close_db_connection()
        return result_set

    def __get_primary_key_of_table(self, pk_name=str, table_name=str, where=str, where_args=dict):
        """
        Retrieves the primary key of the given entry in the given table

        Arguments:
            pk_name (str): The name of the primary key to retrieve
            table_name (str): The table to retrieve the primary key from
            where (str):
                | The WHERE clause string (use ```%(where_args_key)s``` for values).
                |  E.g: 'WHERE username = %(username)s',
                |  and where_args = {'username': 'lucas'}
            where_args (dict): Dictionary with values for the where clause

        See:
            | ```MySQLCursor.fetchall()```
            | ```MySQLdb.cursors.DictCursor```

        Throws:
            ValueError: Throws exception if any of the passed values are empty (None)

        Returns:
            long: Primary key based on passed WHERE statement || ```PRIMARY_KEY_NOT_FOUND```

        """
        error_msg = None
        if pk_name is None:
            error_msg = "Primary key name cannot be empty!"
        elif table_name is None:
            error_msg = "Table name cannot be empty!"
        elif where is None or not where_args:
            error_msg = "Where clause and where arguments cannot be empty!"
        # was there any errors?
        if error_msg is not None:
            raise ValueError(error_msg)
        # everything okay, attempt to retrieve and return primary key
        primary_key = self.PRIMARY_KEY_NOT_FOUND
        try:
            query = "SELECT " + pk_name + " FROM " + table_name + " " + where + ";"
            cursor = self.__get_db_cursor()
            cursor.execute(query, where_args)
            result_set = cursor.fetchone()
            if result_set is None:
                return self.PRIMARY_KEY_NOT_FOUND
            primary_key = result_set[pk_name]
        except MySQLdb.Error as err:
            print("MySQLdb.Error (GET PK): %s", err)
        return primary_key

    def __get_db_cursor(self):
        """
        Returns a cursor for executing database operations.

        See:
            ```MySQLdb.cursors.DictCursor```

        Returns:
            MySQLdb.connect.cursor

        """
        return self.__db.cursor(MySQLdb.cursors.DictCursor)

    def __close_db_connection(self):
        """
        Closes the cursor and the connection to the database
        """
        try:
            self.__db.cursor(MySQLdb.cursors.DictCursor).close()
            self.__db.close()
        except MySQLdb.ProgrammingError as err:
            # if the error is that the connection is already closed, ignore it
            if str(err) == "closing a closed connection":
                pass
            else:
                print("MySQLdb.ProgrammingError: %s", err)
        except MySQLdb.Error as err:
            print("MySQLdb.Error (Close Connection): %s", err)
