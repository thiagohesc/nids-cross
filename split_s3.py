def sql_path(path) -> str:
    return str(path).replace("'", "''")


def count_rows(con, file_path: str) -> int:
    """Conta quantas linhas existem no arquivo."""
    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(file_path)}')
        """
    ).fetchone()[0]


def count_rows_label(con, file_path: str, split_name: str):
    """Conta quantas linhas existem para cada classe Label."""
    rows = con.execute(
        f"""
        SELECT
            "Label",
            COUNT(*) AS total
        FROM read_parquet('{sql_path(file_path)}')
        GROUP BY "Label"
        ORDER BY "Label"
        """
    ).fetchall()

    print(f"{split_name}: {rows}")


def criar_split(
    con,
    source_file: str,
    target_file: str,
    split_name: str,
    condition: str,
    train_size: float,
    val_size: float,
    random_seed: int,
):
    """Monta o SQL para criar arquivos: treino, validação ou teste."""

    sql = f"""
    COPY (
        WITH numbered AS (
            SELECT
                *,
                row_number() OVER () AS __row_id
            FROM read_parquet('{sql_path(source_file)}')
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY "Label"
                    ORDER BY hash(__row_id, {random_seed})
                ) AS __label_rn,
                count(*) OVER (
                    PARTITION BY "Label"
                ) AS __label_total
            FROM numbered
        ),
        limits AS (
            SELECT
                *,
                floor(__label_total * {train_size}) AS __train_max,
                floor(__label_total * {train_size + val_size}) AS __val_max
            FROM ranked
        )
        SELECT *
        EXCLUDE (
            __row_id,
            __label_rn,
            __label_total,
            __train_max,
            __val_max
        )
        FROM limits
        WHERE {condition}
    )
    TO '{sql_path(target_file)}'
    (FORMAT PARQUET, COMPRESSION 'snappy');
    """

    print(f"Gerando {split_name}: {target_file}")
    con.execute(sql)

    total = count_rows(con, target_file)
    print(f"{split_name}: {total:,} linhas")


def split_dataset(
    con,
    dataset_key: str,
    dataset_name: str,
    source_file: str,
    output_path: str,
    train_size: float = 0.60,
    val_size: float = 0.15,
    random_seed: int = 42,
):
    train_file = f"{output_path}/split/{dataset_key}/{dataset_name}_train.parquet"
    val_file = f"{output_path}/split/{dataset_key}/{dataset_name}_val.parquet"
    test_file = f"{output_path}/split/{dataset_key}/{dataset_name}_test.parquet"

    print(f"\nDataset: {dataset_key} ({dataset_name})")
    print(f"Origem: {source_file}")

    total = count_rows(con, source_file)
    print(f"Total limpo: {total:,} linhas")

    criar_split(
        con=con,
        source_file=source_file,
        target_file=train_file,
        split_name="treino",
        condition="__label_rn <= __train_max",
        train_size=train_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    criar_split(
        con=con,
        source_file=source_file,
        target_file=val_file,
        split_name="validacao",
        condition="__label_rn > __train_max AND __label_rn <= __val_max",
        train_size=train_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    criar_split(
        con=con,
        source_file=source_file,
        target_file=test_file,
        split_name="teste",
        condition="__label_rn > __val_max",
        train_size=train_size,
        val_size=val_size,
        random_seed=random_seed,
    )

    print("Distribuicao por Label:")
    count_rows_label(con, train_file, "treino")
    count_rows_label(con, val_file, "validacao")
    count_rows_label(con, test_file, "teste")
    print("-" * 60)
