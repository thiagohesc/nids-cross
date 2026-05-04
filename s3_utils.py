import duckdb

from config import S3_ENDPOINT, S3_URL_STYLE, load_s3_credentials


def conf_s3(con):
    """Aplica credenciais e parametros S3 em uma conexao DuckDB"""
    access_key, secret_key = load_s3_credentials()

    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_endpoint=?;", [S3_ENDPOINT])
    con.execute("SET s3_url_style=?;", [S3_URL_STYLE])
    con.execute("SET s3_access_key_id=?;", [access_key])
    con.execute("SET s3_secret_access_key=?;", [secret_key])


def conectar_duckdb_s3():
    """Cria uma conexao DuckDB ja configurada para acessar o S3 do contabo"""
    con = duckdb.connect()
    conf_s3(con)
    return con
