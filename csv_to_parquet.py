def gerar_amostra(
    con,
    input_path: str,
    output_path: str,
    sample_rows: int = 40000,
    sample_seed: int = 42,
    delimiter: str = ",",
    header: bool = True,
):
    """Gera um CSV de amostra para um dataset bruto no S3"""

    print(f"Gerando amostra {input_path} -> {output_path}")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{input_path}')
            USING SAMPLE {sample_rows} ROWS (reservoir, {sample_seed})
        )
        TO '{output_path}'
        (HEADER {str(header).upper()}, DELIMITER '{delimiter}');
        """
    )

    print(f"Amostra criada: {output_path}")
    print("-" * 50)


def converter_parquet(
    con,
    input_path: str,
    output_path: str,
    compression: str = "snappy",
):
    """Converte o CSV bruto de um dataset para parquet comprimido."""

    print(f"Convertendo {input_path} -> {output_path}")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM '{input_path}'
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION '{compression}');
        """
    )

    print(f"Terminou: {output_path}")
    print("-" * 50)
