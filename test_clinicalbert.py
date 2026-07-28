from transformers import AutoTokenizer, AutoModel
import torch


# ClinicalBERT model
model_name = "emilyalsentzer/Bio_ClinicalBERT"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


print("Loading ClinicalBERT model...")

model = AutoModel.from_pretrained(
    model_name
)


model.eval()


# Example clinical note

text = """
Patient presents with chest pain,
shortness of breath, and history of
coronary artery disease.
"""


# Convert text into tokens

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)


print("Creating clinical embedding...")


with torch.no_grad():

    outputs = model(**inputs)



# CLS token embedding

embedding = outputs.last_hidden_state[:,0,:]


print("Embedding shape:")
print(embedding.shape)
print("Embeddings saved!")

# Train ClinicalBERT Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


print("Training ClinicalBERT classifier...")


bert_classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)


bert_classifier.fit(
    X_train_embeddings,
    y_train
)


print("Classifier training complete!")


bert_predictions = bert_classifier.predict(
    X_test_embeddings
)


bert_accuracy = accuracy_score(
    y_test,
    bert_predictions
)


print("ClinicalBERT Accuracy:", bert_accuracy)


print(
    classification_report(
        y_test,
        bert_predictions
    )
)


joblib.dump(
    bert_classifier,
    "clinicalbert_classifier.pkl"
)


print("ClinicalBERT classifier saved!")