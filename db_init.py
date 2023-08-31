import sqlite3

conn = sqlite3.connect('violations.db')

conn.execute('''CREATE TABLE violations
(id INTEGER PRIMARY KEY AUTOINCREMENT,
violations          INT NOT NULL,
warnings            INT NOT NULL,
created_at          TIMESTAMP,
image               CHAR(100) NOT NULL);''')

conn.commit()
conn.close()
