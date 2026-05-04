from config import RAW_BUCKET, RAW_CSV_FILES, PROCESSED_BUCKET, DATASETS, FEATURES
from s3_utils import conectar_duckdb_s3
from csv_to_parquet import gerar_amostra, converter_parquet
from preprocess import exportar_dataset_limpo, sql_path
from split_s3 import split_dataset
from train import treinar_dataset


def preparar_dataset():
    # Primeira parte:
    # Para os datasets: { bot, cicids, ton, unsw }
    # 1 - Cria amostras do dataset original e salva no s3
    # 2 - Converte os arquivos originais do dataset em parquet e salva no s3
    con = conectar_duckdb_s3()

    for dataset_name in RAW_CSV_FILES:
        input_path = f"{RAW_BUCKET}/{RAW_CSV_FILES[dataset_name]}"

        sample_output = f"{RAW_BUCKET}/samples/{dataset_name}_sample.csv"
        parquet_output = f"{RAW_BUCKET}/parquet/{dataset_name}.parquet"

        gerar_amostra(
            con,
            input_path=input_path,
            output_path=sample_output,
            sample_rows=40000,
            sample_seed=42,
        )

        converter_parquet(
            con,
            input_path=input_path,
            output_path=parquet_output,
            compression="snappy",
        )

    print("Execução terminou!")


def limpar_dataset():
    # Segunda parte:
    # Para os datasets: { bot, cicids, ton, unsw }
    # 1 - Cria novos arquivos com apenas features usadas e limpas
    con = conectar_duckdb_s3()

    for dataset_file in DATASETS.values():
        input_file = f"{RAW_BUCKET}/parquet/{dataset_file}.parquet"
        output_file = f"{PROCESSED_BUCKET}/clean/{dataset_file}_clean.parquet"

        print(f"\nAnalisando dataset ORIGINAL: {input_file}")

        # Antes
        rows_before = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(input_file)}')"
        ).fetchone()[0]

        dist_before = con.execute(
            f"""
            SELECT "Label", COUNT(*) AS total
            FROM read_parquet('{sql_path(input_file)}')
            GROUP BY "Label"
            ORDER BY "Label"
            """
        ).fetchall()

        print(f"Linhas antes: {rows_before:,}")
        print("Distribuição antes:", dist_before)

        # Limpeza
        exportar_dataset_limpo(
            con,
            input_file,
            output_file,
            FEATURES,
        )

        print(f"\nAnalisando dataset LIMPO: {output_file}")

        # Depois
        rows_after = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(output_file)}')"
        ).fetchone()[0]

        dist_after = con.execute(
            f"""
            SELECT "Label", COUNT(*) AS total
            FROM read_parquet('{sql_path(output_file)}')
            GROUP BY "Label"
            ORDER BY "Label"
            """
        ).fetchall()

        print(f"Linhas depois: {rows_after:,}")
        print("Distribuição depois:", dist_after)

        # Resultado da limpeza
        removed = rows_before - rows_after
        pct_removed = (removed / rows_before) * 100 if rows_before > 0 else 0

        print("\nImpacto da limpeza:")
        print(f"Removidas: {removed:,} linhas")
        print(f"% removido: {pct_removed:.2f}%")

        print("-" * 60)

    print("Execução terminou!")


def separar_dataset():
    # Terceira parte:
    # Para os datasets: { bot, cicids, ton, unsw }
    # 1 - Faz a serparação treino, validação e teste usando duckdb
    con = conectar_duckdb_s3()

    TRAIN_SIZE = 0.60
    VAL_SIZE = 0.15
    RANDOM_SEED = 42

    for dataset_key, dataset_name in DATASETS.items():
        source_file = f"{PROCESSED_BUCKET}/clean/{dataset_name}_clean.parquet"

        split_dataset(
            con=con,
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            source_file=source_file,
            output_path=PROCESSED_BUCKET,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            random_seed=RANDOM_SEED,
        )

    print("Execução terminou!")


def treinar_modelo():
    DATASET_TREINO = "cicids"
    con = conectar_duckdb_s3()
    treinar_dataset(con, DATASET_TREINO)


def main():
    # preparar_dataset()
    # limpar_dataset()
    # separar_dataset()
    treinar_modelo()
    # nohup python3 /opt/apps/nids/main.py > main.log 2>&1 &


if __name__ == "__main__":
    main()
