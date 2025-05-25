import mysql.connector
from mysql.connector import Error
import os

def init_database():
    """
    Initialize the database and tables if they don't exist.
    """
    try:
        # Connect to MySQL server without specifying a database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Read the SQL schema file
            with open('db_schema.sql', 'r') as f:
                sql_schema = f.read()

            # First create the database
            cursor.execute("CREATE DATABASE IF NOT EXISTS sentiment_data")
            cursor.execute("USE sentiment_data")

            # Execute each SQL statement separately
            # Split by semicolon but keep CREATE TABLE statements together
            current_statement = ""
            for line in sql_schema.split('\n'):
                line = line.strip()
                if not line or line.startswith('--'):  # Skip empty lines and comments
                    continue

                current_statement += line + " "

                if line.endswith(';'):
                    # Execute the complete statement
                    try:
                        if current_statement.strip():
                            cursor.execute(current_statement)
                    except Exception as e:
                        print(f"Error executing SQL statement: {e}")
                        print(f"Statement: {current_statement}")
                    current_statement = ""

            connection.commit()
            print("Database and tables initialized successfully.")

            # Close the connection
            cursor.close()
            connection.close()

            return True
    except Error as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database()
