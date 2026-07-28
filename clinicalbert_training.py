import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Load dataset

df = pd.read_csv(
    "clinicalbert_training_dataset.csv"
)


print(df.head())
print(df.shape)

# Encode specialty labels

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("Number of specialties:", len(label_encoder.classes_))
print(label_encoder.classes_[:10])
# Separate features and labels

X = df["clean_text"]
y = df["medical_specialty"]


# Encode specialty labels

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("Number of specialties:", len(label_encoder.classes_))
print(label_encoder.classes_[:10])
import pandas as pd

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# Load Bio_ClinicalBERT

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


print("ClinicalBERT loaded!")

import numpy as np

print("Loading saved ClinicalBERT embeddings...")


X_train_embeddings = np.load(
    "X_train_embeddings.npy"
)

X_test_embeddings = np.load(
    "X_test_embeddings.npy"
)


print("Embeddings loaded!")
print("Training shape:", X_train_embeddings.shape)
print("Testing shape:", X_test_embeddings.shape)

# Train Logistic Regression on ClinicalBERT embeddings
import numpy as np

np.save(
    "y_test.npy",
    y_test
)

print("Embeddings saved!")
import numpy as np

# Train Logistic Regression using ClinicalBERT embeddings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


bert_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


bert_model.fit(
    X_train_embeddings,
    y_train
)


bert_predictions = bert_model.predict(
    X_test_embeddings
)


print(
    "ClinicalBERT Accuracy:",
    accuracy_score(y_test, bert_predictions)
)


print(
    classification_report(
        y_test,
        bert_predictions
    )
)
joblib.dump(
    bert_model,
    "clinicalbert_classifier.pkl"
)

print("ClinicalBERT classifier saved!")
import joblib

joblib.dump(
    bert_model,
    "clinicalbert_classifier.pkl"
)

joblib.dump(
    label_encoder,
    "clinicalbert_label_encoder.pkl"
)

print("ClinicalBERT model saved!")

