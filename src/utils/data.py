import streamlit as st
import psycopg2
from psycopg2.extensions import connection


@st.cache_resource(hash_funcs={psycopg2.extensions.connection: id})
def init_connection(host: str, port: str, user_name: str, password: str, database_name: str):
    """
    Initializes and caches a connection to a PostgreSQL database.

    This function attempts to establish a connection to a PostgreSQL database using the provided
    connection parameters. If the connection is successful, it returns the connection object.
    If there is an error while establishing the connection, an error message is printed, and
    the function returns None.

    Args:
        host (str): The hostname or IP address of the PostgreSQL server.
        port (str): The port number on which the PostgreSQL server is listening.
        user_name (str): The username used to authenticate with the PostgreSQL server.
        password (str): The password used to authenticate with the PostgreSQL server.
        database_name (str): The name of the database to connect to.

    Returns:
        psycopg2.extensions.connection: A connection object to the PostgreSQL database, or None
                                        if the connection could not be established.
    """
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database_name,
            user=user_name,
            password=password
        )
        return connection
    except Exception as e:
        print(f"Failed to establish connection: {e}")
        return None

def run_query(query: str, connection: psycopg2.extensions.connection):
    """
    Executes a SQL query against the provided database connection.

    This function attempts to execute the given SQL query using the provided database connection.
    If the connection is None, an error message is displayed, and the function returns None.
    The results of the query are fetched and returned. The results are cached for 10 minutes.

    Args:
        query (str): The SQL query to be executed.
        connection (psycopg2.extensions.connection): The database connection object.

    Returns:
        list: The results of the executed query as a list of tuples, or None if an error occurs
              or the connection is not established.
    """    
    if connection is None:
        st.error("Failed to establish database connection.")
        return None  # Return None if connection is not established
    try:
        with connection.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None