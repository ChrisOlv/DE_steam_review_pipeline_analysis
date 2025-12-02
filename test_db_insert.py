"""
Script de test de connexion et insertion d'une ligne dans une base Azure SQL Database.

Requirements:
    pip install pyodbc
    Installer le pilote ODBC 17 pour SQL Server sur votre machine.

Configuration:
    Exportez les variables d'environnement avant d'exécuter le script, par exemple sous Linux/macOS :
        export DB_SERVER="votre_serveur.database.windows.net"
        export DB_NAME="votre_base"
        export DB_USER="votre_utilisateur"
        export DB_PASSWORD="votre_mot_de_passe"
    Sous Windows (PowerShell) :
        $Env:DB_SERVER="votre_serveur.database.windows.net"
        $Env:DB_NAME="votre_base"
        $Env:DB_USER="votre_utilisateur"
        $Env:DB_PASSWORD="votre_mot_de_passe"

Usage:
    python test_db_insert.py
"""

import os
import pyodbc

def main():
    # Lecture des variables d'environnement
    server   = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    driver   = "{ODBC Driver 17 for SQL Server}"

    if not all([server, database, username, password]):
        raise EnvironmentError("Veuillez définir DB_SERVER, DB_NAME, DB_USER et DB_PASSWORD")

    # Chaîne de connexion ODBC
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
    )

    # Connexion et insertion
    with pyodbc.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            # Création d'une table test si elle n'existe pas
            cursor.execute("""
                IF NOT EXISTS (
                    SELECT * FROM sys.tables WHERE name = 'test_table'
                )
                CREATE TABLE test_table (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    data NVARCHAR(200),
                    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
                )
            """)
            conn.commit()

            # Insertion d'une ligne de test
            test_value = "Hello, Azure SQL!"
            cursor.execute("INSERT INTO test_table (data) VALUES (?)", test_value)
            conn.commit()

            print(f"✅ Ligne insérée dans test_table : '{test_value}'")

if __name__ == "__main__":
    main()