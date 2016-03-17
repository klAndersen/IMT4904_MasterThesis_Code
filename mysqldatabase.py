
import MySQLdb

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
    __TBL_POSTHISTORY = "PostHistory"
    __TBL_POSTS = "Posts"
    __TBL_TAGS = "Tags"
    __TBL_VOTES = "Votes"
    # "Constant" values: Primary keys
    __PK_IS_ID = "Id"
    __TBL_BADGES_ID = "UserId"

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

    def __select_all_records_from_tables(self, table_list, limit=1000):
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
                    query += ", %s"
            # add semi-colon at the end of the query
            query += "LIMIT=" + str(limit) + ";"
            # run query and get the results
            cursor.execute(query % tuple(table_list))
            result_set = cursor.fetchall()
        except MySQLdb.Error as err:
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
