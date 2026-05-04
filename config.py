from pathlib import Path

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

S3_ENDPOINT = "usc1.contabostorage.com"
S3_URL_STYLE = "path"

RAW_BUCKET = "s3://bronze/nids-dataset/raw"
PROCESSED_BUCKET = "s3://bronze/nids-dataset/processed"

DATASETS = {
    "bot": "nf-bot-iot-v3",
    "cicids": "nf-cicids2018-v3",
    "ton": "nf-ton-iot-v3",
    "unsw": "nf-unsw-nb15-v3",
}

RAW_CSV_FILES = {
    "nf-bot-iot-v3": "NF-BoT-IoT-v3/data/NF-BoT-IoT-v3.csv",
    "nf-cicids2018-v3": "NF-CICIDS2018-v3/data/NF-CICIDS2018-v3.csv",
    "nf-ton-iot-v3": "NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv",
    "nf-unsw-nb15-v3": "NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv",
}

# Lista de colunas que serão mantidas no dataset limpo
FEATURES = [
    # Features de protocolo
    "PROTOCOL",
    "L7_PROTO",
    # Features de volume de tráfego
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    # Features de tempo/duração do fluxo
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    # Flags TCP, úteis para identificar comportamento
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    # TTL mínimo e máximo, relacionados ao comportamento de rede/hops
    "MIN_TTL",
    "MAX_TTL",
    # Tamanho dos pacotes do fluxo
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    # Taxas e throughput
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    # Retransmissões, que podem indicar perda, anomalia ou ataque
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    # Distribuição dos pacotes por faixa de tamanho
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    # IAT: Inter Arrival Time
    # Representa o intervalo entre pacotes, útil para capturar padrão temporal
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    # Features relacionadas a ICMP
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    # Label alvo:
    # 0 = normal
    # 1 = ataque
    "Label",
]


# Lista documental das features não usadas
FEATURES_DROP = [
    "IPV4_SRC_ADDR",  # Evita o modelo memorizar IP de origem
    "IPV4_DST_ADDR",  # Evita o modelo memorizar IP de destino
    "L4_SRC_PORT",  # Porta de origem pode ser muito específica do ambiente
    "L4_DST_PORT",  # Porta de destino pode induzir overfitting
    "FLOW_START_MILLISECONDS",  # Timestamp absoluto não generaliza entre datasets
    "FLOW_END_MILLISECONDS",  # Timestamp absoluto não generaliza entre datasets
    "TCP_WIN_MAX_IN",  # Pode depender da infraestrutura/rede
    "TCP_WIN_MAX_OUT",  # Pode depender da infraestrutura/rede
    "DNS_QUERY_ID",  # Geralmente ruído/esparso
    "DNS_QUERY_TYPE",  # Depende muito do perfil DNS do dataset
    "DNS_TTL_ANSWER",  # Muito variável entre ambientes
    "FTP_COMMAND_RET_CODE",  # Só seria útil em tráfego FTP relevante
    "Attack",  # Remove a categoria textual/multiclasse
]


def selected_dataset_names(dataset_key):
    """Converte um alias de dataset em uma lista."""
    if dataset_key == "all":
        return list(DATASETS.values())
    return [DATASETS[dataset_key]]


def load_s3_credentials():
    """Carrega as credenciais S3 do arquivo .env da raiz do projeto."""
    load_dotenv(dotenv_path=ENV_PATH)

    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY")

    if not access_key or not secret_key:
        raise RuntimeError("S3_ACCESS_KEY e S3_SECRET_KEY faltando em .env")

    return access_key, secret_key
