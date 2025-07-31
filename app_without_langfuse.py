import streamlit as st
import pandas as pd
import datetime
import json
from pycaret.regression import load_model, predict_model
from openai import OpenAI
from langfuse import Langfuse

# --- Inicjalizacja Langfuse ---
langfuse = Langfuse()

# --- Wczytanie modelu ---
MODEL_NAME = 'model_polmaraton'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

model = get_model()

# --- UI aplikacji ---
st.title("🏃‍♂️ Przewidywanie czasu półmaratonu")

# --- Klucz OpenAI ---
if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = ""

openai_key = st.text_input("🔑 Wprowadź klucz OpenAI API:", type="password", value=st.session_state["openai_key"])

if not openai_key:
    st.warning("⚠️ Wprowadź klucz API.")
    st.stop()

client = OpenAI(api_key=openai_key)
st.session_state["openai_key"] = openai_key

# --- Model OpenAI i dane użytkownika ---
model_choice = st.selectbox("🤖 Wybierz model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
input_text = st.text_area("📝 Opisz siebie:", height=150)

if st.button("🔮 Oblicz przewidywany czas") and input_text:
    prompt = f"""
    Na podstawie opisu użytkownika zwróć dane w formacie JSON:
    - Płeć (M lub K)
    - Rok (np. 2024)
    - Rocznik (np. 1987)
    - Kategoria wiekowa' (np. M30, K40)

    Opis użytkownika:
    \"{input_text}\"

    Zwróć wyłącznie poprawny JSON.
    """

    try:
        # --- Rejestracja spanu dla promptu ---
        span_prompt = langfuse.span(name="generowanie-promptu", input={"input_text": input_text, "prompt": prompt})
        try:
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "Zwróć tylko poprawny JSON bez komentarza."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            response_text = response.choices[0].message.content.strip()
            span_prompt.end(output={"response_text": response_text}, status="success")
        except Exception as e:
            span_prompt.end(output={"error": str(e)}, status="error")
            raise

        # --- Surowa odpowiedź ---
        st.text_area("🧪 Odpowiedź modelu:", value=response_text, height=200)

        try:
            if response_text.startswith("```json"):
                 response_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
            elif response_text.startswith("```"):
                 response_text = response_text.strip().removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            st.error(f"❌ Błąd parsowania JSON: {e}")
            raise

        df_user = pd.DataFrame([parsed])
        st.subheader("📋 Dane wejściowe")
        st.write(df_user)

        # --- Predykcja ---
        span_pred = langfuse.span(name="predykcja", input=parsed)
        try:
            prediction = predict_model(model, data=df_user)

            if "prediction_label" not in prediction.columns:
                raise ValueError("Brak kolumny 'prediction_label'")

            seconds = prediction["prediction_label"].iloc[0]
            time_str = str(datetime.timedelta(seconds=int(seconds)))

            st.success(f"✅ Przewidywany czas ukończenia biegu: {time_str}")

            span_pred.end(output={
                "czas": time_str,
                "sekundy": seconds,
                "model": model_choice
            }, status="success")
        except Exception as e:
            span_pred.end(output={"error": str(e)}, status="error")
            raise

    except Exception as e:
        st.error(f"❌ Błąd główny: {repr(e)}")
