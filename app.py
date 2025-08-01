import streamlit as st
import pandas as pd
import datetime
import json
from dotenv import load_dotenv
import os
from pycaret.regression import load_model, predict_model
from openai import OpenAI
from langfuse import Langfuse

# --- Załaduj dane z .env ---
load_dotenv()

# --- Inicjalizacja Langfuse z .env ---
langfuse = Langfuse()

# --- Wczytanie modelu ---
MODEL_NAME = 'model_polmaraton'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

model = get_model()

# --- UI ---
st.title("🏃‍♂️ Przewidywanie czasu półmaratonu")

if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = ""

openai_key = st.text_input("🔑 Wprowadź klucz OpenAI API:", type="password", value=st.session_state["openai_key"])
if not openai_key:
    st.warning("⚠️ Wprowadź klucz API.")
    st.stop()

st.session_state["openai_key"] = openai_key
client = OpenAI(api_key=openai_key)

model_choice = st.selectbox("🤖 Wybierz model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
input_text = st.text_area("📝 Opisz siebie:", height=150)

if st.button("🔮 Oblicz przewidywany czas") and input_text:
    trace = langfuse.trace(name="polmaraton", input={"opis": input_text})
    try:
        prompt = f"""
        Na podstawie opisu użytkownika zwróć dane w formacie JSON:
        - Płeć (M lub K)
        - Rok (np. 2024)
        - Rocznik (np. 1987)
        - Kategoria wiekowa (np. M30, K40)

        Opis użytkownika:
        \"{input_text}\"

        Zwróć wyłącznie poprawny JSON.
        """

        # --- Generowanie JSON ---
        span1 = trace.span(name="generowanie-jsona", input={"prompt": prompt})
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "Zwróć tylko poprawny JSON bez komentarza."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw_output = response.choices[0].message.content.strip()
        span1.output = raw_output
        span1.status = "success"
        span1.end()

        st.text_area("🧪 Odpowiedź modelu:", value=raw_output, height=200)

        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_output)
        df_user = pd.DataFrame([parsed])
        st.subheader("📋 Dane wejściowe")
        st.write(df_user)

        # --- Predykcja ---
        span2 = trace.span(name="predykcja", input=parsed)
        prediction = predict_model(model, data=df_user)

        if "prediction_label" not in prediction.columns:
            span2.output = {"błąd": "Brak prediction_label"}
            span2.status = "error"
            span2.end()
            trace.output = {"błąd": "Brak prediction_label"}
       
            st.error("❌ Brak kolumny 'prediction_label'.")
            st.stop()

        seconds = prediction["prediction_label"].iloc[0]
        time_str = str(datetime.timedelta(seconds=int(seconds)))
        span2.output = {"czas": time_str, "sekundy": seconds}
        span2.status = "success"
        span2.end()

        trace.output = {
            "json_od_modelu": parsed,
            "czas": time_str,
            "sekundy": seconds
        }
        

        st.success(f"✅ Przewidywany czas ukończenia biegu: {time_str}")

    except Exception as e:
        trace.output = {"błąd": str(e)}
        trace.status = "error"
        trace.end()
        st.error(f"❌ Błąd główny: {str(e)}")




