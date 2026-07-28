import streamlit as st
import joblib
import numpy as np
import pandas as pd


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Clinical NLP Specialty Classifier",
    page_icon="🏥",
    layout="wide"
)


# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        "clinical_nlp_model.pkl"
    )

    return tfidf, model


tfidf, model = load_model()



# -------------------------
# Specialty Name Cleanup
# -------------------------

specialty_names = {

    "Consult - History and Phy.":
        "Clinical Consultation",

    "Cardiovascular / Pulmonary":
        "Cardiovascular & Pulmonary",

    "Orthopedic":
        "Orthopedics"

}


def clean_specialty(name):

    return specialty_names.get(
        name,
        name
    )



# -------------------------
# Sidebar
# -------------------------

st.sidebar.title(
    "🏥 Clinical NLP Project"
)

st.sidebar.write(
"""
### Current Model

📝 TF-IDF

🤖 Logistic Regression


### Future Upgrade

🧠 Bio_ClinicalBERT

☁️ Azure ML Endpoint
"""
)


# -------------------------
# Main Page
# -------------------------

st.title(
    "🏥 Clinical NLP Medical Specialty Classifier"
)


st.write(
"""
This application analyzes clinical notes
and predicts the most likely medical specialty
using natural language processing.
"""
)


st.warning(
"""
⚠️ This application is an educational AI prototype.
It is not intended for clinical diagnosis or patient care.
"""
)



# -------------------------
# Example Notes
# -------------------------

examples = {

"Cardiology":
"""
Patient presents with chest pain,
shortness of breath, and history of
coronary artery disease.
Cardiac evaluation performed.
""",

"Neurology":
"""
Patient reports seizures and headaches.
Neurological examination performed.
MRI brain ordered.
""",

"Gastroenterology":
"""
Patient presents with abdominal pain.
History of gastrointestinal symptoms.
Endoscopy performed.
"""

}



choice = st.selectbox(
    "Choose Example Clinical Note",
    list(examples.keys())
)



clinical_note = st.text_area(
    "Clinical Note",
    value=examples[choice],
    height=250
)



# -------------------------
# Prediction
# -------------------------

if st.button("🔍 Analyze Clinical Note"):


    if clinical_note.strip():


        features = tfidf.transform(
            [clinical_note]
        )


        probabilities = model.predict_proba(
            features
        )[0]


        classes = model.classes_


        results = pd.DataFrame({

            "Specialty": classes,

            "Probability": probabilities

        })


        results = results.sort_values(
            "Probability",
            ascending=False
        )


        top_prediction = results.iloc[0]


        prediction = clean_specialty(
            top_prediction["Specialty"]
        )



        # Main Prediction

        st.success(
            f"🏥 Predicted Specialty: {prediction}"
        )


        st.info(
            f"📊 Confidence: {top_prediction['Probability']:.2%}"
        )



        st.divider()



        # Top 3

        st.subheader(
            "Top 3 Possible Specialties"
        )


        for _, row in results.head(3).iterrows():

            name = clean_specialty(
                row["Specialty"]
            )


            probability = row["Probability"]


            st.write(
                f"**{name}**"
            )


            st.progress(
                float(probability)
            )


            st.write(
                f"Probability: {probability:.2%}"
            )



        st.divider()



        # Model Information

        st.subheader(
            "Model Information"
        )


        st.write(
"""
**Natural Language Processing Pipeline**

Clinical Text

↓

TF-IDF Vectorization

↓

Logistic Regression Classifier

↓

Medical Specialty Prediction


**Future Version**

Bio_ClinicalBERT + Azure Machine Learning
"""
        )


    else:

        st.warning(
            "Please enter a clinical note."
        )