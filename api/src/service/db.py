import sqlite3
import numpy as np
import io

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

class GeneratorZsDb:
    def __init__(self, path: str):
        self.db_filename = path + '/ZS.db'
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        self.connection = sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        self.create_sql_table_seeds()

    def create_sql_table_seeds(self):
        print("Connecting to database at: " + self.db_filename)        

        self.connection.cursor().execute(
            "CREATE TABLE IF NOT EXISTS ZS ("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
            "Z array,"
            "UNIQUE (Z)"
            ")"
        )

        self.connection.commit()

    

    def insert_z(self, z:np.ndarray):
        self.connection = sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES)

        self.connection.cursor().execute(
            "INSERT INTO ZS (Z) VALUES(?)",
            (z,))

        self.connection.commit()

    def fetch_z_by_id(self, id: int):
        cursor = self.connection.cursor()

        cursor.execute('SELECT Z FROM ZS WHERE ID == ?', (id,))

        return cursor.fetchone()[0]

    def fetch_all(self):
        cursor = self.connection.cursor()

        cursor.execute('SELECT * FROM ZS')

        return cursor.fetchall()
