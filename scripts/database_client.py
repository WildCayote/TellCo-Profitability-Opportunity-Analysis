import pandas as pd
import psycopg2

class DB_Client:
    """
    A client for interacting with a PostgreSQL database.

    This class establishes a connection to a PostgreSQL database and can be used to execute queries
    and interact with the database. It uses `psycopg2` to handle the database connection.

    Attributes:
        host (str): The hostname or IP address of the database server.
        user_name (str): The username used to authenticate to the database.
        password (str): The password used to authenticate to the database.
        port (str): The port number the database is listening on.
        database_name (str): The name of the specific database to connect to.
    """

    def __init__(self, host: str, user_name: str, password: str, port: str, database_name: str):
        """
        Initializes the DB_Client with the given connection details.

        Args:
            host (str): The hostname or IP address of the database server.
            user_name (str): The username used to authenticate to the database.
            password (str): The password used to authenticate to the database.
            port (str): The port number the database is listening on.
            database_name (str): The name of the database to connect to.
        """
        self.host = host
        self.user_name = user_name
        self.password = password
        self.port = port
        self.database_name = database_name
        self.connection = self.__establish_connection()

    def __establish_connection(self):
        """
        Establishes a connection to the PostgreSQL database.

        This method attempts to establish a connection to the database using the provided
        connection details. It uses `psycopg2.connect` to create the connection. If the
        connection fails, the method returns `None`.

        Returns:
            connection: A `psycopg2` connection object if successful, or `None` if the connection
                        fails.
        
        Raises:
            Exception: If there is an error in establishing the connection.
        """
        try:
            connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database_name,
                user=self.user_name,
                password=self.password
            )
            return connection
        except Exception as e:
            print(f"Failed to establish connection: {e}")
            return None

    def execute_query(self, query: str):
        """
        Executes a SQL query on the connected PostgreSQL database.

        This method takes a SQL query as input, executes it, and returns the result as a pandas DataFrame.

        Args:
            query (str): The SQL query to be executed.

        Returns:
            pandas.DataFrame: A DataFrame containing the query result if successful.
            None: Returns `None` if the query execution fails.
        
        Raises:
            Exception: If there is an error while executing the query.
        """
        try:
            response = pd.read_sql_query(sql=query, con=self.connection)
            return response
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return None

    def dump_data(self, table: str = 'xdr_data'):
        """
        Retrieves all data from a specified table in the PostgreSQL database.

        This method generates a SQL `SELECT * FROM {table}` query and fetches the data from the given
        table, using the `execute_query` method.

        Args:
            table (str): The name of the table from which data will be selected. Defaults to 'xdr_data'.

        Returns:
            pandas.DataFrame: A DataFrame containing all the rows from the specified table if successful.
            None: Returns `None` if the query execution fails.
        """
        query = f"SELECT * FROM {table}"
        return self.execute_query(query=query)
