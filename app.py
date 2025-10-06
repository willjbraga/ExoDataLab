# app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="DataLab SpaceApps – Exoplanet ML", page_icon="🚀", layout="wide")


# 🔧 Aumenta a área do cabeçalho da sidebar e o tamanho do logo
st.markdown("""
<style>
/* dá mais espaço ao cabeçalho da sidebar */
[data-testid="stSidebarHeader"] { 
  padding: 16px 16px 8px !important; 
  min-height: 96px !important;
}
/* aumenta o logo acima do menu */
[data-testid="stSidebarHeader"] img {
  height: 250px !important;      /* ajuste: 64 / 72 / 96 */
  width: auto !important;
  object-fit: contain !important;
}
@media (max-width: 900px){
  [data-testid="stSidebarHeader"] img { height: 64px !important; }
}
</style>
""", unsafe_allow_html=True)

st.logo("assets/sidebar_header.png", size = 'large' )  # coloque seu título/subtítulo no PNG

# H1 global (aparece em todas as páginas)
st.title("DataLab SpaceApps – Exoplanet AI")

ROOT = Path(__file__).parent.resolve()  # .../exodatalab



nav = st.navigation({
    "Home":  [st.Page(str(ROOT / "homepage.py"), title="HomePage", icon=":material/home:")],
    "Data":  [st.Page(str(ROOT / "results.py"),  title="Results",   icon=":material/insights:")],
})
nav.run()
