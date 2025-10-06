# RESULTS (única fonte de verdade = confusion_matrix)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import streamlit as st

ss  = st.session_state
out = ss["results"]

y_true = pd.Series(out["True"],  name="Real")
y_pred = pd.Series(out["Preds"], name="Predito")

# ordem de classes vinda do y_true e completada pelo y_pred
labels = list(pd.unique(y_true))
labels += [c for c in pd.unique(y_pred) if c not in labels]

# -------- MATRIZ (única fonte) --------
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Tabela a partir da cm (garante consistência com o gráfico)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

st.subheader("Matriz de Confusão — Tabela")
st.dataframe(cm_df)  # nada de crosstab

# Checagem de sanidade (opcional)
# assert cm.sum() == len(y_true) == cm_df.values.sum()

# -------- Gráfico Matplotlib (mesma cm) --------
#st.subheader("Matriz de Confusão — Gráfico (Matplotlib)")
#fig, ax = plt.subplots(figsize=(6, 6))
#ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
#    cmap="Blues", values_format="d", ax=ax, colorbar=False
#)
#ax.set_title("Matriz de Confusão", fontsize=14)
#plt.tight_layout()
#st.pyplot(fig)

# -------- Versão interativa Plotly (0 branco, >0 azul) --------
with st.expander("Versão interativa (Plotly)"):
    blues = px.colors.sequential.Blues
    zero_white = [(0.0, "#ffffff")] + [(i/(len(blues)-1), c) for i, c in enumerate(blues)]
    fig2 = px.imshow(
        cm_df.values,
        x=[f"pred:{c}" for c in cm_df.columns],
        y=[f"real:{r}" for r in cm_df.index],
        text_auto="d",
        color_continuous_scale=zero_white,
        zmin=0, zmax=float(cm_df.values.max() or 1),
        labels={"x": "Predito", "y": "Real", "color": "Contagem"},
        aspect="auto",
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig2)
