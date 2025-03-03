import streamlit as st
import pandas as pd
import sqlite3

# 🔌 Connexion à la base MLflow
DB_FILE = "mlflow.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_FILE)
    df_metrics = pd.read_sql_query("SELECT * FROM metrics", conn)
    df_params = pd.read_sql_query("SELECT * FROM params", conn)
    df_runs = pd.read_sql_query("SELECT * FROM runs", conn)
    conn.close()
    return df_metrics, df_params, df_runs

df_metrics, df_params, df_runs = load_data()

# 🎨 Interface Streamlit
st.title("📊 Dashboard MLflow - Expérimentations")

# 📌 Afficher les métriques
st.subheader("📈 Metrics enregistrées")
st.dataframe(df_metrics)

# 📌 Afficher les paramètres
st.subheader("⚙️ Paramètres des modèles")
st.dataframe(df_params)

# 📌 Meilleurs résultats
best_runs = df_metrics.sort_values(by="value", ascending=True).head(10)
st.subheader("🏆 Top 10 Meilleurs Modèles (selon la métrique)")
st.dataframe(best_runs)

# 📌 Téléchargement des résultats
st.download_button("📥 Télécharger les métriques (CSV)", df_metrics.to_csv(index=False), "mlflow_metrics.csv")
st.download_button("📥 Télécharger les paramètres (CSV)", df_params.to_csv(index=False), "mlflow_params.csv")
