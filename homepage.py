# HomePage.py
import csv
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
#st.set_page_config(
#    page_title="DataLab SpaceApps – Exoplanet ML",
#    page_icon="🚀",
#    layout="wide",
#)


ss = st.session_state
ss.setdefault("raw_df", None)
ss.setdefault("clean_df", None)
ss.setdefault("schema_ok", False)
ss.setdefault("status_msg", "Aguardando dados")
ss.setdefault("log", [])

# ------------------------------------------------------------
# Esquema esperado (nomes canônicos + sinônimos + tipo)
# Inclui o exemplo citado: "Per Koi_Period" → float
# ------------------------------------------------------------
REQUIRED_SCHEMA = {
    "orbital_period": {
        "type": "float",
        "aliases": ["Per Koi_Period", "Koi_Period", "koi_period", "orbital_period", "period", "per"],
    },
    "transit_duration": {
        "type": "float",
        "aliases": ["koi_duration", "transit_duration", "duration", "dur"],
    },
    "planet_radius": {
        "type": "float",
        "aliases": ["koi_prad", "planet_radius", "pl_rade", "prad", "radius"],
    },
    "star_teff": {
        "type": "float",
        "aliases": ["koi_steff", "st_teff", "star_teff", "teff"],
    },
    "star_logg": {
        "type": "float",
        "aliases": ["koi_slogg", "st_logg", "star_logg", "logg"],
    },
    "snr": {
        "type": "float",
        "aliases": ["koi_snr", "snr"],
    },
}
CANONICAL_ORDER = list(REQUIRED_SCHEMA.keys())

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _lower_strip(s: str) -> str:
    return s.strip().lower().replace("\n", " ")

def sniff_is_csv(upload) -> bool:
    name_ok = upload.name.lower().endswith(".csv")
    try:
        sample = upload.read(4096).decode("utf-8", errors="ignore")
        upload.seek(0)
        csv.Sniffer().sniff(sample)
        sniff_ok = True
    except Exception:
        sniff_ok = False
    return name_ok and sniff_ok

def read_csv_any(upload):
    try:
        upload.seek(0)
        return pd.read_csv(upload)              # tenta separador padrão (,)
    except Exception:
        upload.seek(0)
        return pd.read_csv(upload, sep=";")     # fallback para ;

def standardize_columns(df: pd.DataFrame):
    col_map = {}
    lower_map = {_lower_strip(c): c for c in df.columns}
    missing = []
    for canon, spec in REQUIRED_SCHEMA.items():
        found = None
        for alias in spec["aliases"]:
            key = _lower_strip(alias)
            if key in lower_map:
                found = lower_map[key]
                break
        if found is None:
            missing.append(canon)
        else:
            col_map[found] = canon
    df2 = df.rename(columns=col_map)
    return df2, missing, col_map

def coerce_types(df: pd.DataFrame):
    issues = []
    df2 = df.copy()
    for col, spec in REQUIRED_SCHEMA.items():
        if col not in df2.columns:
            continue
        if spec["type"] == "float":
            before_non_null = df2[col].notna().sum()
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
            after_non_null = df2[col].notna().sum()
            bad = before_non_null - after_non_null
            if bad > 0:
                issues.append(f"Coluna '{col}': {bad} valor(es) inválido(s) para float → convertidos em NaN.")
        else:
            issues.append(f"Tipo '{spec['type']}' ainda não implementado para '{col}'.")
    return df2, issues

def validate_schema(df: pd.DataFrame):
    log = []

    # 1) Padronizar nomes
    df_std, missing, mapping = standardize_columns(df)
    if mapping:
        mapped_str = ", ".join([f"'{src}'→'{dst}'" for src, dst in mapping.items()])
        log.append(f"Mapeamento de colunas: {mapped_str}.")
    if missing:
        log.append("Colunas obrigatórias ausentes: " + ", ".join([f"'{m}'" for m in missing]) + ".")
        return df_std, False, log

    # 2) Converter tipos
    df_typed, type_issues = coerce_types(df_std)
    log.extend(type_issues)

    # 3) Checar NaNs nas obrigatórias (permitimos NaN; tratamento pode ser em outra etapa)
    na_counts = df_typed[CANONICAL_ORDER].isna().sum()
    na_total = int(na_counts.sum())
    if na_total > 0:
        brk = ", ".join([f"{c}:{int(n)}" for c, n in na_counts.items() if n > 0])
        log.append(f"Valores ausentes após conversão → {na_total} ({brk}).")

    ok = len(missing) == 0
    return df_typed, ok, log

def sample_df():
    return pd.DataFrame({
        "Per Koi_Period": [1.0, 1.6, 1.2],
        "transit_duration": [0.1, 1.1, 1.4],
        "planet_radius": [1.2, 1.2, 1.5],
        "star_teff": [5700, 5700, 5700],
        "star_logg": [4.4, 4.3, 4.5],
        "snr": [4.4, 5.4, 5.6],
    })

# ------------------------------------------------------------
# Sidebar (menu)
# ------------------------------------------------------------
#st.sidebar.title("DataLab SpaceApps")
#st.sidebar.caption("Machine Learning for exoplanets detection")
#page = st.sidebar.radio("Navigation", ["Home", "Treinar Modelo", "Classificar", "Métricas", "Sobre"], index=0)
#page = st.navigation(
#    {
#        "Home": [st.Page("homepage.py", title = "HomePage", icon = ":material/home:")],
#        "Data": [st.Page("results.py", title = "Results", icon = ":material/insights:")]
#    }
#)

# ------------------------------------------------------------
# Cabeçalho
# ------------------------------------------------------------
#st.title("DataLab SpaceApps – Exoplanet AI")
st.caption("Carregue um CSV no padrão exigido; o app detecta colunas, valida tipos e prepara o pipeline.")

m1, m2, m3 = st.columns(3)
m1.metric("Acurácia (validação)", "—")
m2.metric("F1-Score", "—")
m3.metric("Status do modelo", ss["status_msg"])

# ------------------------------------------------------------
# Conteúdo por página (a Home realiza apenas validação)
# ------------------------------------------------------------
#if page != "Início":
#    st.info("Esta é a Home. Use as páginas **Treinar Modelo** e **Classificar** (outros arquivos) para o restante do fluxo.")
#    st.stop()

# -------- Início (Home) --------
st.subheader("Dados de entrada")

# Card com campos obrigatórios
with st.expander("📋 Campos obrigatórios do CSV (padrão exigido)", expanded=True):
    schema_rows = []
    for canon, spec in REQUIRED_SCHEMA.items():
        schema_rows.append({
            "campo (canônico)": canon,
            "tipo": spec["type"],
            "sinônimos aceitos": ", ".join(spec["aliases"]),
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)
    st.caption("Os nomes do seu arquivo podem ser qualquer um dos **sinônimos**; eles serão normalizados para o nome canônico.")

# Upload + Preview
upload = st.file_uploader("Carregue um CSV seguindo o padrão acima.", type=["csv"])
if upload is not None:
    if not sniff_is_csv(upload):
        st.error("Arquivo não reconhecido como CSV válido (.csv).")
    else:
        try:
            df = read_csv_any(upload)
            ss["raw_df"] = df.copy()
        except Exception as e:
            st.error(f"Falha ao ler CSV: {e}")

preview = ss["raw_df"] if ss["raw_df"] is not None else sample_df()
st.dataframe(preview.head(), use_container_width=True)

st.write("""
## Hyperparameter Configuration
""")
iterations = st.slider(label = "Iterations",min_value = 0, max_value = 1000, value=500)
depth = st.slider(label = "Depth",min_value = 0, max_value = 20, step = 1, value = 6 )
learning_rate = st.slider (label = "Learning Rate", min_value = 0.00001, max_value = 0.001, step = 0.00001, format="%0.5f", value = 0.001)


# Opções (somente sinalização visual; tratadas de fato em outras páginas)
#c1, c2, c3 = st.columns(3)
#detect_numeric = c1.checkbox("Detectar colunas numéricas", value=False)
#treat_missing  = c2.checkbox("Tratar valores ausentes", value=False)
#normalize_feat = c3.checkbox("Normalizar features", value=False)

# Ações
a, b, c = st.columns(3)
if a.button("Limpar & Validar", use_container_width=True):
    base = ss["raw_df"] if ss["raw_df"] is not None else preview
    df_checked, ok, v_log = validate_schema(base)

    #if detect_numeric:
    #    v_log.append("Sinalização: detectar colunas numéricas (aplicação completa no pré-processo).")
    #if treat_missing:
    #    v_log.append("Sinalização: tratar valores ausentes (aplicação completa no pré-processo).")
    #if normalize_feat:
    #    v_log.append("Sinalização: normalizar features (aplicação na etapa de modelagem).")

    ss["clean_df"] = df_checked
    ss["schema_ok"] = ok
    ss["log"] = v_log
    ss["status_msg"] = "Dados validados" if ok else "Schema inválido"
    st.success("CSV validado e padronizado." if ok else "CSV inválido para o padrão. Verifique o log.")

if b.button("Treinar Modelo", use_container_width=True):
    st.info("Treino ocorre em outra página/arquivo (ex.: TreinarModelo.py) usando st.session_state['clean_df'].")

if c.button("Classificar", use_container_width=True):
    st.info("Classificação ocorre em outra página/arquivo (ex.: Classificar.py) após o treino.")

# Log
with st.expander("Log de validação"):
    if ss["log"]:
        for line in ss["log"]:
            st.write("• " + line)
    else:
        st.write("Tipos detectados, valores ausentes, escalonamento pronto.")

st.caption("Feito com ❤️ por DataLab — NASA Space Apps 2025")
