import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="ClinicalBERT Clinical Note Classifier",
    page_icon="🧠",
    layout="wide"
)


# -------------------------
# Load ClinicalBERT
# -------------------------

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



# -------------------------
# Load Classifier
# -------------------------

@st.cache_resource
def load_classifier():

    classifier = joblib.load(
        "clinicalbert_classifier.pkl"
    )


    label_encoder = joblib.load(
        "clinicalbert_label_encoder.pkl"
    )


    return classifier, label_encoder



classifier, label_encoder = load_classifier()



# -------------------------
# Generate Embeddings
# -------------------------

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



# -------------------------
# Sidebar
# -------------------------

st.sidebar.title(
    "🧠 Clinical NLP Project"
)


st.sidebar.markdown(
"""
## Project

**Clinical Note Specialty Classification**

### Dataset

Medical Transcriptions Dataset

- 4,964 clinical notes
- 40 medical specialties


### Model Architecture

Clinical Text

↓

Bio_ClinicalBERT

↓

768-Dimensional Embedding

↓

Logistic Regression

↓

Specialty Prediction


### Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Streamlit


### Future Deployment

☁️ Azure Machine Learning

🔗 FHIR / EHR Integration

🏥 Healthcare NLP Automation
"""
)



# -------------------------
# Main Page
# -------------------------

st.title(
    "🧠 ClinicalBERT Clinical Note Specialty Classification"
)



st.write(
"""
### Project Goal

This application demonstrates how transformer-based
Natural Language Processing models can analyze
unstructured clinical documentation and classify
medical notes into their associated specialty.
"""
)



st.warning(
"""
⚠️ Educational AI prototype only.

This system is not intended for clinical diagnosis,
medical decision-making, or patient care.
"""
)



# -------------------------
# Input
# -------------------------

clinical_note = st.text_area(
    "Enter Clinical Note",
    height=250,
    value="""
Patient presents with chest pain,
shortness of breath, and history of
coronary artery disease.
"""
)



# -------------------------
# Prediction
# -------------------------

if st.button(
    "🔍 Analyze Clinical Note"
):


    if clinical_note.strip():


        embedding = generate_embedding(
            clinical_note
        )



        prediction_number = classifier.predict(
            embedding
        )[0]



        specialty = label_encoder.inverse_transform(
            [prediction_number]
        )[0]



        probabilities = classifier.predict_proba(
            embedding
        )[0]



        classes = label_encoder.inverse_transform(
            np.arange(len(probabilities))
        )



        results = pd.DataFrame({

            "Specialty": classes,

            "Probability": probabilities

        })



        results = results.sort_values(
            "Probability",
            ascending=False
        )



        confidence = results.iloc[0]["Probability"]



        # -------------------------
        # Prediction Result
        # -------------------------

        st.success(
            f"🏥 Predicted Specialty: {specialty}"
        )



        st.info(
            f"🧠 ClinicalBERT Confidence: {confidence:.2%}"
        )



        st.caption(
"""
Confidence represents the model's probability estimate,
not clinical certainty. Predictions should be reviewed
by qualified healthcare professionals.
"""
        )



        st.divider()



        # -------------------------
        # Top Predictions
        # -------------------------

        st.subheader(
            "Top 3 Possible Specialties"
        )


        for _, row in results.head(3).iterrows():


            st.markdown(
                f"### {row['Specialty']}"
            )


            st.progress(
                float(row["Probability"])
            )


            st.write(
                f"Confidence: {row['Probability']:.2%}"
            )



        st.divider()



        # -------------------------
        # Model Performance
        # -------------------------

        st.subheader(
            "Model Performance"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "ClinicalBERT Test Accuracy",
                "19.3%"
            )


        with col2:

            st.metric(
                "Clinical Notes",
                "4,964"
            )


        st.caption(
"""
Performance is influenced by dataset size,
40 specialty categories, and class imbalance.
Future improvements include fine-tuning ClinicalBERT
and expanding clinical datasets.
"""
        )



        st.divider()



        # -------------------------
        # Model Comparison
        # -------------------------

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


            "Strength":[

                "Fast baseline approach",

                "Understands clinical language context"

            ]

        })


        st.table(
            comparison
        )



        st.divider()



        # -------------------------
        # Pipeline
        # -------------------------

        st.subheader(
            "Clinical NLP Pipeline"
        )


        st.write(
"""
Clinical Note

↓

Text Processing

↓

Bio_ClinicalBERT

↓

768-Dimensional Embedding

↓

Logistic Regression Classifier

↓

Medical Specialty Prediction


### Healthcare Applications

☁️ Azure Machine Learning Deployment

🔗 FHIR / EHR Integration

🏥 Automated Clinical Document Routing

📄 Medical Information Retrieval
"""
        )


    else:

        st.warning(
            "Please enter a clinical note."
        )