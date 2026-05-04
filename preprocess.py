def to_string(col: str) -> str:
    return f'"{col}"'


def sql_path(path) -> str:
    return str(path).replace("'", "''")


def exportar_dataset_limpo(
    con,
    input_path: str,
    output_path: str,
    features: list[str],
):
    """Remove featureso que não usadas salva no s3."""

    num_features = [col for col in features if col != "Label"]

    cast_features = ",\n                ".join(
        f"TRY_CAST({to_string(col)} AS DOUBLE) AS {to_string(col)}"
        for col in num_features
    )

    # Artigo: descartar registros incompletos
    condicao_null = " AND\n            ".join(
        f"{to_string(col)} IS NOT NULL" for col in num_features + ["Label"]
    )

    # Gera condições para remover valores infinitos com erro
    condicao_finite = " AND\n            ".join(
        f"isfinite({to_string(col)})" for col in num_features
    )

    sql = f"""
    COPY (
        WITH typed AS (
            SELECT
                {cast_features},
                TRY_CAST("Label" AS TINYINT) AS "Label"
            FROM read_parquet('{sql_path(input_path)}')
        )
        SELECT *
        FROM typed
        WHERE
            "Label" IN (0, 1)
            AND {condicao_null}
            AND {condicao_finite}
    )
    TO '{sql_path(output_path)}'
    (FORMAT PARQUET, COMPRESSION 'snappy');
    """

    print(f"Limpando dataset: {input_path}")
    con.execute(sql)
    print(f"Arquivo criado: {output_path}")
    return output_path
