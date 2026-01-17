import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000/").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
HISTORY_URL = f"{API_URL}/history"

st.set_page_config(page_title="Prédiction réussite scolaire", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("Prédiction de la réussite scolaire")
st.caption(
    "Saisissez les caractéristiques de l’élève, puis obtenez une note finale prédite et une décision réussite/échec."
)


YES_NO_LABELS = ["Oui", "Non"]
YES_NO_MAP = {"Oui": "yes", "Non": "no"}
YES_NO_REV = {v: k for k, v in YES_NO_MAP.items()}

SCHOOL_LABELS = ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"]
SCHOOL_MAP = {"Gabriel Pereira (GP)": "GP", "Mousinho da Silveira (MS)": "MS"}
SCHOOL_REV = {v: k for k, v in SCHOOL_MAP.items()}

REASON_LABELS = [
    "Programme / options proposées",
    "Proximité du domicile",
    "Réputation de l’établissement",
    "Autre raison",
]
REASON_MAP = {
    "Programme / options proposées": "course",
    "Proximité du domicile": "home",
    "Réputation de l’établissement": "reputation",
    "Autre raison": "other",
}
REASON_REV = {v: k for k, v in REASON_MAP.items()}


def predict(payload: dict):
    r = requests.post(
        PREDICT_URL, params={"session_id": st.session_state.session_id}, json=payload, timeout=30
    )
    if r.status_code != 200:
        raise RuntimeError(f"Erreur API ({r.status_code}) : {r.text}")
    return r.json()


def get_history(limit=50, session_id=None):
    params = {"limit": limit}
    if session_id:
        params["session_id"] = session_id
    r = requests.get(HISTORY_URL, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Erreur API ({r.status_code}) : {r.text}")
    return r.json()


tab1, tab2 = st.tabs(["Formulaire de prédiction", "Historique des inférences"])

with tab1:
    st.subheader("Données de l'élève")

    with st.form("student_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        
        # Colonne 1
        
        with c1:
            school_label = st.selectbox(
                "Établissement (GP/MS)",
                SCHOOL_LABELS,
                index=0,
                help="GP = Gabriel Pereira, MS = Mousinho da Silveira.",
            )
            school = SCHOOL_MAP[school_label]

            age = st.number_input(
                "Âge (15–22 ans)",
                min_value=15,
                max_value=22,
                value=17,
                help="Âge de l’élève au moment de l’observation.",
            )

            reason_label = st.selectbox(
                "Raison du choix de l’école",
                REASON_LABELS,
                index=0,
                help="Motif principal du choix de l’établissement.",
            )
            reason = REASON_MAP[reason_label]

            nursery_label = st.selectbox(
                "A suivi la maternelle ?",
                YES_NO_LABELS,
                index=0,
                help="Indique si l’élève a fréquenté la maternelle.",
            )
            nursery = YES_NO_MAP[nursery_label]

            traveltime = st.slider(
                "Temps de trajet domicile–école (1–4)",
                1,
                4,
                2,
                help="1 = très court, 4 = long.",
            )

        
        # Colonne 2
        
        with c2:
            studytime = st.slider(
                "Temps d’étude hebdomadaire (1–4)",
                1,
                4,
                2,
                help="1 = faible, 4 = élevé.",
            )

            failures = st.slider(
                "Nombre d’échecs scolaires (0–4)",
                0,
                4,
                0,
                help="Nombre de fois où l’élève a redoublé/échoué.",
            )

            schoolsup_label = st.selectbox(
                "Soutien scolaire (école) ?",
                YES_NO_LABELS,
                index=1,
                help="Soutien éducatif fourni par l’établissement.",
            )
            schoolsup = YES_NO_MAP[schoolsup_label]

            famsup_label = st.selectbox(
                "Soutien familial ?",
                YES_NO_LABELS,
                index=0,
                help="Soutien éducatif fourni par la famille.",
            )
            famsup = YES_NO_MAP[famsup_label]

            paid_label = st.selectbox(
                "Cours particuliers de maths ?",
                YES_NO_LABELS,
                index=1,
                help="Cours payants en dehors de l’école (maths).",
            )
            paid = YES_NO_MAP[paid_label]

        
        # Colonne 3
        
        with c3:
            activities_label = st.selectbox(
                "Activités extra‑scolaires ?",
                YES_NO_LABELS,
                index=1,
                help="Participation à des activités en dehors de l’école.",
            )
            activities = YES_NO_MAP[activities_label]

            higher_label = st.selectbox(
                "Souhaite faire des études supérieures ?",
                YES_NO_LABELS,
                index=0,
                help="Intention de poursuivre des études après le secondaire.",
            )
            higher = YES_NO_MAP[higher_label]

            freetime = st.slider(
                "Temps libre après l’école (1–5)",
                1,
                5,
                3,
                help="1 = très faible, 5 = très élevé.",
            )

            goout = st.slider(
                "Sorties avec amis (1–5)",
                1,
                5,
                3,
                help="1 = rarement, 5 = très souvent.",
            )

            absences = st.number_input(
                "Nombre d’absences",
                min_value=0,
                max_value=200,
                value=2,
                help="Nombre total d’absences.",
            )

        st.markdown("### Notes trimestrielles")
        cc1, cc2 = st.columns(2)
        with cc1:
            G1 = st.number_input(
                "Note – Trimestre 1 (0–20)",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                help="Note obtenue au trimestre 1 (décimales autorisées).",
            )
        with cc2:
            G2 = st.number_input(
                "Note – Trimestre 2 (0–20)",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                help="Note obtenue au trimestre 2 (décimales autorisées).",
            )

        submitted = st.form_submit_button("Soumettre")

    if submitted:
       
        payload = {
            "school": school,
            "age": int(age),
            "reason": reason,
            "nursery": nursery,
            "traveltime": int(traveltime),
            "studytime": int(studytime),
            "failures": int(failures),
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "freetime": int(freetime),
            "goout": int(goout),
            "absences": int(absences),
            "G1": float(G1),
            "G2": float(G2),
        }

        try:
            res = predict(payload)
            predicted = res["predicted_G3"]
            decision = res["decision"]
            model_version = res.get("model_version") or res.get("mlflow_model_version")

            st.success(f"Note finale prédite (G3) : **{predicted:.2f}/20**")
            if decision == "reussite":
                st.info("Décision : **Réussite de l’année** (>= 10)")
            else:
                st.warning("❌ Décision : **Échec** (< 10)")

            with st.expander("Détails (interprétables)"):
                st.write("- **Règle** : réussite si G3 >= 10")
                st.write("- **Seuil** : 10")
                st.write(f"- **Version modèle** : {model_version}")
                st.write(f"- **Session ID** : {res['session_id']}")

        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Historique")
    colA, colB = st.columns([1, 2])
    with colA:
        limit = st.slider("Nombre d'entrées", 10, 200, 50, 10)
        only_me = st.checkbox("Filtrer sur ma session", value=True)
        refresh = st.button("Rafraîchir")

    if refresh or True:
        try:
            hist = get_history(limit=limit, session_id=st.session_state.session_id if only_me else None)
            if not hist:
                st.info("Aucune inférence enregistrée pour l’instant.")
            else:
                df = pd.DataFrame(hist)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.dataframe(
                    df[["timestamp", "session_id", "predicted_G3", "decision", "model_version"]],
                    use_container_width=True,
                )

                with st.expander("Voir les entrées (inputs_json)"):
                    st.json(hist[0]["inputs_json"])
        except Exception as e:
            st.error(str(e))