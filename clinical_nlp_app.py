import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Clinical NLP Medical Specialty Classification",
    page_icon="🏥",
    layout="wide"
)


# ------------------------------------------------
# Load TF-IDF Model
# ------------------------------------------------

@st.cache_resource
def load_tfidf_model():

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        "clinical_nlp_model.pkl"
    )

    return tfidf, model


tfidf, tfidf_model = load_tfidf_model()



# ------------------------------------------------
# Load ClinicalBERT Model
# ------------------------------------------------

@st.cache_resource
def load_clinicalbert():

    model_name = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    clinicalbert = AutoModel.from_pretrained(
        model_name
    )

    clinicalbert.eval()


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    clinicalbert.to(device)


    return tokenizer, clinicalbert, device



tokenizer, clinicalbert, device = load_clinicalbert()



# ------------------------------------------------
# Load ClinicalBERT Classifier
# ------------------------------------------------

@st.cache_resource
def load_clinicalbert_classifier():

    classifier = joblib.load(
        "clinicalbert_classifier.pkl"
    )


    label_encoder = joblib.load(
        "clinicalbert_label_encoder.pkl"
    )


    return classifier, label_encoder



clinicalbert_model, label_encoder = load_clinicalbert_classifier()



# ------------------------------------------------
# Specialty Cleanup
# ------------------------------------------------

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



# ------------------------------------------------
# ClinicalBERT Embedding Generator
# ------------------------------------------------

def generate_embedding(text):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )


    inputs = {
        key:value.to(device)
        for key,value in inputs.items()
    }


    with torch.no_grad():

        outputs = clinicalbert(
            **inputs
        )


    embedding = (
        outputs.last_hidden_state[:,0,:]
        .cpu()
        .numpy()
    )


    return embedding



# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.title(
    "🏥 Clinical NLP Project"
)


st.sidebar.markdown(
"""
## Models Available

📝 **TF-IDF + Logistic Regression**

Traditional NLP baseline using word frequency features.


🧠 **Bio_ClinicalBERT + Logistic Regression**

Transformer-based model using contextual clinical embeddings.


---

## Dataset

Medical Transcriptions Dataset

- 4,964 Clinical Notes
- 40 Medical Specialties


---

## Deployment Roadmap

☁️ **Azure Machine Learning**

Future goal:
Deploy NLP models to the cloud
for scalable healthcare text analysis.
"""
)



# ------------------------------------------------
# Main Page
# ------------------------------------------------

st.title(
    "🏥 Clinical NLP Medical Specialty Classification"
)


st.write(
"""
This application compares traditional NLP and
transformer-based NLP approaches for classifying
unstructured clinical documentation into medical specialties.
"""
)


st.warning(
"""
⚠️ Educational AI prototype only.

This system is not intended for clinical diagnosis,
medical decision-making, or patient care.
"""
)



# ------------------------------------------------
# Model Selection
# ------------------------------------------------

model_choice = st.selectbox(
    "Select NLP Model",
    [
        "TF-IDF + Logistic Regression",
        "ClinicalBERT + Logistic Regression"
    ]
)



# ------------------------------------------------
# Clinical Note Input
# ------------------------------------------------

clinical_note = st.text_area(
    "Enter Clinical Note",
    height=250,
    value="""
Patient presents with chest pain,
shortness of breath, and history of
coronary artery disease.
"""
)



# ------------------------------------------------
# Prediction
# ------------------------------------------------

if st.button("🔍 Analyze Clinical Note"):


    if clinical_note.strip():


        # -----------------------------
        # TF-IDF Prediction
        # -----------------------------

        if model_choice == "TF-IDF + Logistic Regression":


            features = tfidf.transform(
                [clinical_note]
            )


            probabilities = tfidf_model.predict_proba(
                features
            )[0]


            classes = tfidf_model.classes_


            results = pd.DataFrame({

                "Specialty": classes,

                "Probability": probabilities

            })


            pipeline = """
Clinical Note

↓

TF-IDF Vectorization

↓

Logistic Regression

↓

Medical Specialty Prediction
"""


            model_name = "TF-IDF + Logistic Regression"



        # -----------------------------
        # ClinicalBERT Prediction
        # -----------------------------

        else:


            embedding = generate_embedding(
                clinical_note
            )


            probabilities = clinicalbert_model.predict_proba(
                embedding
            )[0]


            classes = label_encoder.inverse_transform(
                np.arange(len(probabilities))
            )


            results = pd.DataFrame({

                "Specialty": classes,

                "Probability": probabilities

            })


            pipeline = """
Clinical Note

↓

Bio_ClinicalBERT

↓

768-Dimensional Embedding

↓

Logistic Regression

↓

Medical Specialty Prediction
"""


            model_name = "ClinicalBERT + Logistic Regression"



        # Sort results

        results = results.sort_values(
            "Probability",
            ascending=False
        )


        top_prediction = results.iloc[0]


        prediction = clean_specialty(
            top_prediction["Specialty"]
        )


        confidence = top_prediction["Probability"]



        # -----------------------------
        # Results
        # -----------------------------

        st.success(
            f"🏥 Predicted Specialty: {prediction}"
        )


        st.info(
            f"Model: {model_name}\n\nConfidence: {confidence:.2%}"
        )



        st.divider()



        st.subheader(
            "Top 3 Possible Specialties"
        )


        for _, row in results.head(3).iterrows():

            specialty = clean_specialty(
                row["Specialty"]
            )


            st.write(
                f"**{specialty}**"
            )


            st.progress(
                float(row["Probability"])
            )


            st.write(
                f"Probability: {row['Probability']:.2%}"
            )



        st.divider()



        # -----------------------------
        # Comparison
        # -----------------------------

        st.subheader(
            "Traditional NLP vs Transformer NLP"
        )


        comparison = pd.DataFrame({

            "Model":[

                "TF-IDF + Logistic Regression",

                "ClinicalBERT + Logistic Regression"

            ],

            "Representation":[

                "Word frequency features",

                "Contextual clinical embeddings"

            ],

            "Purpose":[

                "Fast baseline approach",

                "Advanced clinical language understanding"

            ]

        })


        st.table(
            comparison
        )



        st.divider()



        st.subheader(
            "Clinical NLP Pipeline"
        )


        st.write(
            pipeline
        )


    else:

        st.warning(
            "Please enter a clinical note."
        )