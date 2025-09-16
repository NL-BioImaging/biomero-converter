import sqlite3


class DBReader:
    """
    Reads and queries a SQLite database, returning results as dictionaries.
    """

    def __init__(self, db_file):
        """
        Initialize DBReader with a database file.

        Args:
            db_file (str): Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = DBReader.dict_factory

    @staticmethod
    def dict_factory(cursor, row):
        """
        Converts a database row to a dictionary.

        Args:
            cursor: SQLite cursor object.
            row: Row data.

        Returns:
            dict: Mapping column names to values.
        """
        dct = {}
        for index, column in enumerate(cursor.description):
            dct[column[0]] = row[index]
        return dct

    def fetch_all(self, query, params=(), return_dicts=True):
        """
        Executes a query and fetches all results.

        Args:
            query (str): SQL query string.
            params (tuple): Query parameters.
            return_dicts (bool): If True, returns list of dicts; else, returns first column values.

        Returns:
            list: Query results.
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        dct = cursor.fetchall()
        if return_dicts:
            values = dct
        else:
            values = [list(row.values())[0] for row in dct]
        return values

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
