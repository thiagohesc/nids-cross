import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tensorflow.keras.models import load_model
from config import DATASETS, PROCESSED_BUCKET, PROJECT_ROOT
from s3_utils import conectar_duckdb_s3


TARGET_LABEL = 1
NN_BATCH_SIZE = 32


def sql_path(path: str) -> str:
    return str(path).replace("'", "''")


def get_test_file(dataset_key: str) -> tuple[str, str]:
    """Monta o path do arquivo de teste para o dataset alvo."""
    dataset_name = DATASETS[dataset_key]
    test_file = f"{PROCESSED_BUCKET}/split/{dataset_key}/{dataset_name}_test.parquet"
    return dataset_name, test_file


def read_parquet(con, file_path: str) -> pd.DataFrame:
    """Lê um arquivo parquet do S3 usando DuckDB."""
    print(f"Lendo: {file_path}")
    return con.execute(
        f"""
        SELECT *
        FROM read_parquet('{sql_path(file_path)}')
        """
    ).fetchdf()


def carregar_modelo(source_key: str):
    """Carrega o modelo treinado e artefatos do dataset de origem."""
    model_dir = PROJECT_ROOT / "models" / source_key
    model = load_model(model_dir / "best_model.keras", compile=False)
    scaler = joblib.load(model_dir / "scaler.pkl")
    pca = joblib.load(model_dir / "pca.pkl")
    features = pd.read_csv(model_dir / "features.csv")["feature"].tolist()
    return model, scaler, pca, features


def split_features(df: pd.DataFrame, features: list[str]):
    """Separa features X e target y mantendo as colunas usadas no treino."""
    check_features = [col for col in features if col not in df.columns]

    if check_features:
        raise ValueError(f"Colunas faltando no dataset alvo: {check_features}")

    x = df[features].astype("float32")
    y = df["Label"].astype("int8")

    return x, y


def avaliar_cross(con, source_key: str, target_key: str) -> dict:
    """Avalia um modelo treinado em outro dataset."""

    print("\n" + "=" * 60)
    print(f"CROSS: {source_key} -> {target_key}")

    # Modelo e pré-processadores treinados no dataset de origem
    model, scaler, pca, features = carregar_modelo(source_key)

    # Dataset de teste usado como alvo da avaliação
    target_dataset_name, target_test_file = get_test_file(target_key)

    df_test = read_parquet(con, target_test_file)

    # Usa apenas as features conhecidas pelo modelo de origem
    x_test, y_test = split_features(df_test, features)

    del df_test

    print(f"Shape teste: {x_test.shape}")
    print("Distribuição Label:")
    print(y_test.value_counts().sort_index())

    # Aplica o mesmo scaler e PCA usados durante o treino
    x_test_scaled = scaler.transform(x_test)
    x_test_pca = pca.transform(x_test_scaled)

    # Predição do modelo no dataset alvo
    y_prob = model.predict(
        x_test_pca,
        batch_size=NN_BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(y_prob, axis=1)
    y_attack_prob = y_prob[:, TARGET_LABEL]

    # Métricas para a classe de ataque
    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=TARGET_LABEL,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=TARGET_LABEL,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=TARGET_LABEL,
        zero_division=0,
    )

    roc_auc = roc_auc_score(y_test, y_attack_prob)

    cm = confusion_matrix(y_test, y_pred)

    # Relatório detalhado por classe
    report_dict = classification_report(
        y_test,
        y_pred,
        digits=6,
        zero_division=0,
        output_dict=True,
    )

    print("\nMétricas:")
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 attack: {f1:.6f}")
    print(f"ROC AUC: {roc_auc:.6f}")

    print("\nMatriz de confusão:")
    print(cm)

    # Artefatos da avaliação cruzada
    output_dir = PROJECT_ROOT / "models" / source_key / "cross"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_name = f"{source_key}_to_{target_key}"

    pd.DataFrame(
        cm,
        index=["real_normal_0", "real_attack_1"],
        columns=["pred_normal_0", "pred_attack_1"],
    ).to_csv(output_dir / f"{pair_name}_confusion_matrix.csv")

    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / f"{pair_name}_classification_report.csv"
    )

    # Registro resumido para comparação entre pares source -> target
    result = {
        "source": source_key,
        "target": target_key,
        "target_dataset_name": target_dataset_name,
        "target_test_file": target_test_file,
        "test_rows": len(y_test),
        "accuracy": acc,
        "precision_attack": precision,
        "recall_attack": recall,
        "f1_attack": f1,
        "roc_auc": roc_auc,
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }

    pd.DataFrame([result]).to_csv(
        output_dir / f"{pair_name}_metrics.csv",
        index=False,
    )

    return result


if __name__ == "__main__":
    con = conectar_duckdb_s3()

    SOURCE = "unsw"
    TARGET = "cicids"

    avaliar_cross(
        con=con,
        source_key=SOURCE,
        target_key=TARGET,
    )
