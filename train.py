import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from config import DATASETS, PROCESSED_BUCKET, PROJECT_ROOT
from f1callback import F1Callback


TARGET_LABEL = 1
RANDOM_STATE = 42
NN_BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
L2_FACTOR = 0.0001
PCA_COMPONENTS = 20
DROPOUT_RATE = 0.15
EARLY_STOP_PATIENCE = 7


def sql_path(path) -> str:
    return str(path).replace("'", "''")


def construir_modelo(input_dim: int) -> Sequential:
    """Cria a rede neural MLP para classificar a label"""

    model = Sequential(
        [
            Dense(
                64,
                activation="relu",
                input_shape=(input_dim,),
                kernel_regularizer=l2(L2_FACTOR),
            ),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(
                32,
                activation="relu",
                kernel_regularizer=l2(L2_FACTOR),
            ),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(
                16,
                activation="relu",
                kernel_regularizer=l2(L2_FACTOR),
            ),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def split_paths(dataset_key: str) -> tuple[str, str, str, str]:
    """Monta os PATHS dos arquivos train, val e test no S3"""

    dataset_name = DATASETS[dataset_key]

    train_file = f"{PROCESSED_BUCKET}/split/{dataset_key}/{dataset_name}_train.parquet"
    val_file = f"{PROCESSED_BUCKET}/split/{dataset_key}/{dataset_name}_val.parquet"
    test_file = f"{PROCESSED_BUCKET}/split/{dataset_key}/{dataset_name}_test.parquet"

    return dataset_name, train_file, val_file, test_file


def read_parquet(con, file_path: str) -> pd.DataFrame:
    print(f"Lendo arquivo: {file_path}")
    return con.execute(
        f"""
        SELECT *
        FROM read_parquet('{sql_path(file_path)}')
        """
    ).fetchdf()


def split_features_label(df: pd.DataFrame):
    """Separa features X e target y."""

    x = df.drop(columns=["Label"]).astype("float32")
    y = df["Label"].astype("int8")

    return x, y


def show_label_distribution(y: pd.Series, split_name: str):
    """Mostra distribuição de Label no split."""
    print(f"\nDistribuição {split_name}:")
    print(y.value_counts().sort_index())


def calcular_pesos(y_train: pd.Series) -> dict[int, float]:
    """Calcula pesos das classes"""

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        0: float(weights[0]),
        1: float(weights[1]),
    }


def treinar_dataset(con, dataset_key: str):
    """Treina o modelo para um DATASET."""

    dataset_name, train_file, val_file, test_file = split_paths(dataset_key)

    output_dir = PROJECT_ROOT / "models" / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_key} ({dataset_name})")
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print(f"Test:  {test_file}")

    train_df = read_parquet(con, train_file)
    val_df = read_parquet(con, val_file)
    test_df = read_parquet(con, test_file)

    x_train, y_train = split_features_label(train_df)
    x_val, y_val = split_features_label(val_df)
    x_test, y_test = split_features_label(test_df)

    del train_df, val_df, test_df

    print(f"\nTreino:    {x_train.shape}")
    print(f"Validação: {x_val.shape}")
    print(f"Teste:     {x_test.shape}")

    show_label_distribution(y_train, "treino")
    show_label_distribution(y_val, "validação")
    show_label_distribution(y_test, "teste")

    feature_columns = x_train.columns.tolist()
    print(f"\nTotal de features antes do PCA: {len(feature_columns)}")

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    print(f"Variância explicada pelo PCA: {explained_variance:.6f}")

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(pca, output_dir / "pca.pkl")

    pd.DataFrame({"feature": feature_columns}).to_csv(
        output_dir / "features.csv",
        index=False,
    )

    class_weight = calcular_pesos(y_train)
    print(f"\nClass weight usado no treino: {class_weight}")

    model = construir_modelo(input_dim=PCA_COMPONENTS)

    callbacks = [
        F1Callback(
            validation_data=(x_val_pca, y_val),
            target_label=TARGET_LABEL,
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor="val_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train_pca,
        y_train,
        validation_data=(x_val_pca, y_val),
        epochs=EPOCHS,
        batch_size=NN_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(output_dir / "model.keras")

    pd.DataFrame(history.history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    print("\nAvaliando no conjunto de teste...")

    y_prob = model.predict(
        x_test_pca,
        batch_size=NN_BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(y_prob, axis=1)
    y_attack_prob = y_prob[:, TARGET_LABEL]

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

    report = classification_report(
        y_test,
        y_pred,
        digits=6,
        zero_division=0,
    )

    print("\nMétricas:")
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision label {TARGET_LABEL}: {precision:.6f}")
    print(f"Recall label {TARGET_LABEL}: {recall:.6f}")
    print(f"F1 label {TARGET_LABEL}: {f1:.6f}")
    print(f"ROC AUC: {roc_auc:.6f}")

    print("\nMatriz:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset_key,
                "dataset_name": dataset_name,
                "train_file": train_file,
                "val_file": val_file,
                "test_file": test_file,
                "target_label": TARGET_LABEL,
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "test_rows": len(y_test),
                "pca_components": PCA_COMPONENTS,
                "pca_explained_variance": explained_variance,
                "accuracy": acc,
                "precision_target": precision,
                "recall_target": recall,
                "f1_target": f1,
                "roc_auc": roc_auc,
                "class_weight_0": class_weight[0],
                "class_weight_1": class_weight[1],
                "nn_batch_size": NN_BATCH_SIZE,
                "epochs_configured": EPOCHS,
                "epochs_trained": len(history.history["loss"]),
                "learning_rate": LEARNING_RATE,
                "l2_factor": L2_FACTOR,
                "dropout_rate": DROPOUT_RATE,
                "early_stop_patience": EARLY_STOP_PATIENCE,
            }
        ]
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)

    pd.DataFrame(
        cm,
        index=["real_normal_0", "real_attack_1"],
        columns=["pred_normal_0", "pred_attack_1"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nArtefatos salvos em: {output_dir}")
