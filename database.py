import sqlite3
import json

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('evaluations.db')
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                candidate_name TEXT,
                score REAL,
                verdict TEXT,
                full_results_json TEXT
            );
        ''')
    except sqlite3.Error as e:
        print(e)

def insert_evaluation(conn, results):
    sql = ''' INSERT INTO evaluations(candidate_name, score, verdict, full_results_json)
              VALUES(?,?,?,?) '''
    c = conn.cursor()
    
    profile = results.get('structured_data', {})
    candidate_name = profile.get('name', results.get('filename', 'Unknown Candidate'))

    score = results.get('score', 0)
    
    verdict = "Low Fit"
    if score >= 75:
        verdict = "High Fit"
    elif score >= 50:
        verdict = "Medium Fit"


    full_results_str = json.dumps(results)

    data = (
        candidate_name,
        score,
        verdict,
        full_results_str
    )
    c.execute(sql, data)
    conn.commit()
    return c.lastrowid

def clear_all_evaluations(conn):
    sql = 'DELETE FROM evaluations'
    c = conn.cursor()
    c.execute(sql)
    conn.commit()
    print("All Evaluations Have Been Cleared.")

def init_db():
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        conn.close()
    else:
        print("Error : Cannot Create The Database Connection.")
