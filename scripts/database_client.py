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
        self.cursor = self.connection.cursor()

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
            if 'SELECT' in query:
                response = pd.read_sql_query(sql=query, con=self.connection)
                return response
            else:
                self.cursor.execute(query=query)
                self.connection.commit()
                return None
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

    def insert_data(self, data: pd.DataFrame, table: str = 'xdr_data'):
        """
        Inserts data into a given table in the PostgreSQL database. The table schema must be created before executing this function.

        Args:
            data (pd.DataFrame): The data we want to push to the table, the key of the dataframe is considered as the primary key for the database
            table (str)L The name of the table we want to push the data to

        Returns:
            int: signifying the success
        """

        try:
            # Convert DataFrame into a list of tuples (needed for psycopg2)
            rows = [tuple(x) for x in data.to_numpy()]

            # Get column names
            columns = ', '.join([f'"{col}"' for col in [data.index.name, *data.columns]])

            # Gather all row values
            values_list = []
            for index, row in data.iterrows():
                values = [str(index)]
                for value in row:
                    if pd.isna(value):
                        values.append('NULL')
                    else:
                        # Escape single quotes by replacing them with double single quotes for SQL
                        escaped_value = str(value).replace("'", "''")
                        values.append(f"{escaped_value}")
                values_list.append(f"({', '.join(values)})")

            # values concatinated
            concatenated_values = ",\n".join(values_list) + ";\n"

            # Create the final SQL query string
            insert_query = f"INSERT INTO {table} ({columns}) VALUES {concatenated_values}"
            
            # execute the query
            result = self.execute_query(query=insert_query)
            
            return result
        
        except Exception as e:
            print(f"Couldn't push data becuase of: {e}")
