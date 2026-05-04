from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class F1Callback(Callback):
    """
    Callback genérico para calcular F1, precision e recall
    de uma classe específica ao final de cada época.
    """

    def __init__(
        self,
        validation_data,
        target_label: int = 1,
        prefix: str = "val",
    ):
        super().__init__()

        # Dados de validação (já pré-processados)
        self.x_val, self.y_val = validation_data

        # Classe alvo
        self.target_label = target_label

        # Prefixo para logs (ex: val, test, etc)
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Predição do modelo
        y_prob = self.model.predict(self.x_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        # Métricas para a classe alvo
        f1 = f1_score(
            self.y_val,
            y_pred,
            pos_label=self.target_label,
            zero_division=0,
        )

        precision = precision_score(
            self.y_val,
            y_pred,
            pos_label=self.target_label,
            zero_division=0,
        )

        recall = recall_score(
            self.y_val,
            y_pred,
            pos_label=self.target_label,
            zero_division=0,
        )

        f1_key = f"{self.prefix}_f1_label_{self.target_label}"
        precision_key = f"{self.prefix}_precision_label_{self.target_label}"
        recall_key = f"{self.prefix}_recall_label_{self.target_label}"

        # Salva no logs (para EarlyStopping / ModelCheckpoint)
        logs[f1_key] = f1
        logs[precision_key] = precision
        logs[recall_key] = recall

        print(
            f" - {f1_key}: {f1:.6f}"
            f" - {precision_key}: {precision:.6f}"
            f" - {recall_key}: {recall:.6f}"
        )
