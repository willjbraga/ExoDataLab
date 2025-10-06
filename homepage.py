# HomePage.py
import csv
import pandas as pd
import streamlit as st
import yaml
from pathlib import Path

from ml_.infer import infer_light 


# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
# st.set_page_config(
#     page_title="DataLab SpaceApps – Exoplanet ML",
#     page_icon="🚀",
#     layout="wide",
# )

CFG_PATH = Path(__file__).resolve().parent / "ml_" / "ml_utils.yaml"

@st.cache_data
def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_cfg(cfg: dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

cfg = load_cfg()
model_cfg = cfg.get("model", {})

ss = st.session_state
ss.setdefault("raw_df", None)
ss.setdefault("status_msg", "Waiting for input")
ss.setdefault("target_col", None)
ss.setdefault("feature_cols", None)
ss.setdefault("X", None)
ss.setdefault("y", None)

# (Opcional) Card informativo dos campos recomendados — não faz validação
REQUIRED_SCHEMA = {
    "orbital_period": {"type": "float", "aliases": ["Per Koi_Period", "Koi_Period", "koi_period", "period", "per"]},
    "transit_duration": {"type": "float", "aliases": ["koi_duration", "transit_duration", "duration", "dur"]},
    "planet_radius": {"type": "float", "aliases": ["koi_prad", "planet_radius", "pl_rade", "prad", "radius"]},
    "star_teff": {"type": "float", "aliases": ["koi_steff", "st_teff", "star_teff", "teff"]},
    "star_logg": {"type": "float", "aliases": ["koi_slogg", "st_logg", "star_logg", "logg"]},
    "snr": {"type": "float", "aliases": ["koi_snr", "snr"]},
}

def read_csv_any(upload):
    # Garante que a leitura comece do início do arquivo
    upload.seek(0)
    try:
        # 1. Tenta ler como uma tabela com espaços como separador (o seu caso)
        return pd.read_csv(upload, engine='python', comment='#')
    except Exception:
        upload.seek(0)
        try:
            # 2. Tenta ler como um CSV padrão (separado por vírgula)
            return pd.read_csv(upload, comment='#')
        except Exception:
            upload.seek(0)
            # 3. Tenta ler como um CSV separado por ponto e vírgula
            return pd.read_csv(upload, sep=';', comment='#')

def sample_df():
    return pd.DataFrame({
        "Per Koi_Period": [1.0, 1.6, 1.2],
        "transit_duration": [0.1, 1.1, 1.4],
        "planet_radius": [1.2, 1.2, 1.5],
        "star_teff": [5700, 5700, 5700],
        "star_logg": [4.4, 4.3, 4.5],
        "snr": [4.4, 5.4, 5.6],
        "koi_disposition": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
    })

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.caption("Upload a CSV; the app only reads and shows the data. Select the training target on a different page.")

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy (validation set)", "—")
m2.metric("F1-Score", "—")
m3.metric("Model status", ss["status_msg"])

st.subheader("Dados de entrada")

# Card informativo (sem validação)
with st.expander("📋 Colunas recomendadas para o dataset (informativo – sem validação)", expanded=False):
    schema_rows = []
    for canon, spec in REQUIRED_SCHEMA.items():
        schema_rows.append({
            "campo (canônico)": canon,
            "tipo": spec["type"],
            "sinônimos aceitos": ", ".join(spec["aliases"]),
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)
    st.caption("Apenas informativo. Seu CSV é aceito como estiver; a padronização/limpeza pode ser feita depois.")

# Upload + Preview
upload = st.file_uploader("Upload a CSV file.", type=["csv"])
if upload is not None:
    try:
        df = read_csv_any(upload)
        ss["raw_df"] = df.copy()
        ss["status_msg"] = "CSV file loaded"
    except Exception as e:
        st.error(f"CSV import failed: {e}")


preview = ss["raw_df"] if ss["raw_df"] is not None else sample_df()
st.dataframe(preview.head(), use_container_width=True)

st.write("## Hyperparameter Configuration")
iterations = st.slider("Iterations", min_value=0, max_value=1000, value=500)
depth = st.slider("Depth", min_value=0, max_value=20, step=1, value=6)
learning_rate = st.slider("Learning Rate", min_value=0.00001, max_value=0.001, step=0.00001, format="%0.5f", value=0.001)


# Target
st.subheader("Target column")
if ss["raw_df"] is None:
    st.info("Upload a CSV to unlock target selection.")
else:
    df_ok = ss["raw_df"]
    cols = list(df_ok.columns)

    # palpites de nomes comuns para target
    guesses = ["koi_disposition", "disposition", "koi_pdisposition", "label", "target", "class", "status"]
    default_target = ss["target_col"] if ss["target_col"] in cols else None
    if default_target is None:
        for g in guesses:
            if g in cols:
                default_target = g
                break
    default_index = cols.index(default_target) if default_target in cols else 0

    target_col = st.selectbox(
        "Select the **target** column:",
        options=cols,
        index=default_index,
        help="This column will serve as the label (y); the rest are features (X)."
    )
    ss["target_col"] = target_col

    name_col = st.selectbox(
        "Select the **planet names** column:",
        options=cols,
        index=default_index,
        help="This column will serve as the planet labels; "
    )
    ss["name_col"] = name_col

    y = df_ok[target_col]
    if pd.api.types.is_numeric_dtype(y):
        st.caption("Resumo do target (numérico):")
        st.dataframe(y.describe().to_frame("estatística"), use_container_width=True)
    else:
        st.caption("Distribuição de classes do target:")
        vc = y.value_counts(dropna=False).to_frame("contagem")
        vc["proporção"] = (vc["contagem"] / len(y)).round(4)
        st.dataframe(vc, use_container_width=True)

    feature_cols = [c for c in df_ok.columns if c != target_col]
    ss["feature_cols"] = feature_cols
    ss["X"] = df_ok[feature_cols]
    ss["y"] = y

    st.success(f"Target definido: **{target_col}**. Use st.session_state['X'], ['y'], ['feature_cols'], ['target_col'].")

df = st.session_state.get("raw_df")
target = ss["target_col"]
print(df)

# Ação (somente Classificar)
if st.button("Classificar", use_container_width=True):
    # manter também na sessão para esta execução/página seguinte
    ss["hp"] = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
    }

    try:
        cfg.setdefault("model", {})
        cfg["model"].update(ss["hp"])
        save_cfg(cfg)
        st.toast("Hiperparâmetros gravados em ml_utils.yaml", icon="✅")
    except Exception as e:
        st.warning(f"Não consegui gravar o YAML ({e}). Usarei os valores em memória.")

    preds, metrics, y_true, labels = infer_light(df, target, upload.name, name_col)
    ss["results"] = {
        "Preds": preds,
        "Metrics" : metrics,
        "True" : y_true,
        "Labels" : labels
    }

    st.switch_page("results.py")


st.caption("Feito com ❤️ por DataLab — NASA Space Apps 2025")
