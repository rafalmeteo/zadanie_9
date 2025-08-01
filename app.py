import streamlit as st
import pandas as pd
import datetime
import json
import joblib

from openai import OpenAI
from langfuse import Langfuse

# --- Inicjalizacja Langfuse z secrets ---
langfuse = Langfuse(
    public_key=st.secrets["LANGFUSE_PUBLIC_KEY"],
    secret_key=st.secrets["LANGFUSE_SECRET_KEY"],
    host=st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# --- Wczytanie modelu ---
MODEL_NAME = 'model_polmaraton.pkl'

@st.cache_data
def get_model():
    return joblib.load(MODEL_NAME)

model = get_model()

# --- UI ---
st.title("🏃‍♂️ Przewidywanie czasu półmaratonu")

# --- Klucz OpenAI ---
if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = ""

openai_key = st.text_input("🔑 Wprowadź klucz OpenAI API:", type="password", value=st.session_state["openai_key"])
st.session_state["openai_key"] = openai_key

if not openai_key:
    st.warning("⚠️ Wprowadź klucz OpenAI API, aby kontynuować.")
    st.stop()

client = OpenAI(api_key=openai_key)

# --- Dane użytkownika ---
model_choice = st.selectbox("🤖 Wybierz model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
input_text = st.text_area("📝 Opisz siebie (wiek, płeć, doświadczenie biegowe):", height=150)

if st.button("🔮 Oblicz przewidywany czas") and input_text:
    trace = langfuse.trace(name="polmaraton", input={"opis": input_text})
    try:
        # --- Prompt do LLM ---
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

        # --- Generowanie JSON przez LLM ---
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

        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        span1.output = raw_output
        span1.status = "success"
        span1.end()

        st.text_area("🧪 Odpowiedź modelu:", value=raw_output, height=200)

        # --- Parsowanie odpowiedzi do DataFrame ---
        parsed = json.loads(raw_output)
        df_user = pd.DataFrame([parsed])

        st.subheader("📋 Dane wejściowe")
        st.write(df_user)

        # --- Predykcja ---
        span2 = trace.span(name="predykcja", input=parsed)
        prediction = model.predict(df_user)

        if len(prediction) == 0:
            raise ValueError("Model nie zwrócił żadnej predykcji.")

        seconds = prediction[0]
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
        st.error(f"❌ Błąd: {str(e)}")
