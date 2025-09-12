# sqlite_patch.py
import sys
import pysqlite3

# Force Chroma to use modern sqlite
sys.modules["sqlite3"] = pysqlite3
