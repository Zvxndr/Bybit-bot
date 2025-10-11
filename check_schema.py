import sqlite3

conn = sqlite3.connect('data/historical_data.db')
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Available tables:')
for table in tables:
    print(f'  {table[0]}')
    
if tables:
    # Get schema for first table
    table_name = tables[0][0]
    cursor = conn.execute(f'PRAGMA table_info({table_name})')
    columns = cursor.fetchall()
    print(f'\nSchema for {table_name}:')
    for col in columns:
        print(f'  {col[1]} ({col[2]})')
        
conn.close()