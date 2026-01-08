from flask import Blueprint, request, jsonify

database_bp = Blueprint('database', __name__)

@database_bp.route("/api/database/test-connection", methods=["POST"])
def test_database_connection():
    try:
        data = request.json
        db_type = data.get('db_type', 'sqlite')
        
        if db_type == 'sqlite':
            import sqlite3
            file_path = data.get('file_path', '')
            if not file_path:
                return jsonify({"success": False, "error": "File path is required"}), 400
            conn = sqlite3.connect(file_path)
            conn.close()
            return jsonify({"success": True, "message": "Connection successful"})
            
        elif db_type == 'postgresql':
            try:
                import psycopg2
            except ImportError:
                return jsonify({"success": False, "error": "psycopg2 not installed. Install with: pip install psycopg2-binary"}), 400
            
            host = data.get('host', 'localhost')
            port = data.get('port', 5432)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            
            if not database:
                return jsonify({"success": False, "error": "Database name is required"}), 400
            
            conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
            conn.close()
            return jsonify({"success": True, "message": "Connection successful"})
            
        elif db_type == 'mysql':
            try:
                import mysql.connector
            except ImportError:
                return jsonify({"success": False, "error": "mysql-connector-python not installed. Install with: pip install mysql-connector-python"}), 400
            
            host = data.get('host', 'localhost')
            port = data.get('port', 3306)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            
            if not database:
                return jsonify({"success": False, "error": "Database name is required"}), 400
            
            conn = mysql.connector.connect(host=host, port=port, database=database, user=username, password=password)
            conn.close()
            return jsonify({"success": True, "message": "Connection successful"})
        
        return jsonify({"success": False, "error": f"Unsupported database type: {db_type}"}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@database_bp.route("/api/database/fetch-tables", methods=["POST"])
def fetch_database_tables():
    try:
        data = request.json
        db_type = data.get('db_type', 'sqlite')
        tables = []
        
        if db_type == 'sqlite':
            import sqlite3
            file_path = data.get('file_path', '')
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
        elif db_type == 'postgresql':
            import psycopg2
            host = data.get('host', 'localhost')
            port = data.get('port', 5432)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
        elif db_type == 'mysql':
            import mysql.connector
            host = data.get('host', 'localhost')
            port = data.get('port', 3306)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = mysql.connector.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
        
        return jsonify({"success": True, "tables": tables})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@database_bp.route("/api/database/preview-data", methods=["POST"])
def preview_database_data():
    try:
        data = request.json
        db_type = data.get('db_type', 'sqlite')
        fetch_mode = data.get('fetch_mode', 'table')
        preview_rows = data.get('preview_rows', 10)
        
        if fetch_mode == 'table':
            table_name = data.get('table_name', '')
            if not table_name:
                return jsonify({"success": False, "error": "Table name is required"}), 400
            query = f"SELECT * FROM {table_name} LIMIT {preview_rows}"
        else:
            query = data.get('query', '')
            if not query:
                return jsonify({"success": False, "error": "SQL query is required"}), 400
            if 'LIMIT' not in query.upper():
                query = f"{query} LIMIT {preview_rows}"
        
        if db_type == 'sqlite':
            import sqlite3
            file_path = data.get('file_path', '')
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
            
        elif db_type == 'postgresql':
            import psycopg2
            host = data.get('host', 'localhost')
            port = data.get('port', 5432)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
            
        elif db_type == 'mysql':
            import mysql.connector
            host = data.get('host', 'localhost')
            port = data.get('port', 3306)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = mysql.connector.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
        
        rows_list = [list(row) for row in rows]
        return jsonify({"success": True, "headers": headers, "rows": rows_list})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@database_bp.route("/api/database/load-data", methods=["POST"])
def load_database_data():
    try:
        data = request.json
        db_type = data.get('db_type', 'sqlite')
        fetch_mode = data.get('fetch_mode', 'table')
        
        if fetch_mode == 'table':
            table_name = data.get('table_name', '')
            if not table_name:
                return jsonify({"success": False, "error": "Table name is required"}), 400
            query = f"SELECT * FROM {table_name}"
        else:
            query = data.get('query', '')
            if not query:
                return jsonify({"success": False, "error": "SQL query is required"}), 400
        
        if db_type == 'sqlite':
            import sqlite3
            file_path = data.get('file_path', '')
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
            
        elif db_type == 'postgresql':
            import psycopg2
            host = data.get('host', 'localhost')
            port = data.get('port', 5432)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
            
        elif db_type == 'mysql':
            import mysql.connector
            host = data.get('host', 'localhost')
            port = data.get('port', 3306)
            database = data.get('database', '')
            username = data.get('username', '')
            password = data.get('password', '')
            conn = mysql.connector.connect(host=host, port=port, database=database, user=username, password=password)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            conn.close()
        
        rows_list = [list(row) for row in rows]
        return jsonify({
            "success": True,
            "headers": headers,
            "rows": rows_list,
            "row_count": len(rows_list),
            "column_count": len(headers)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
