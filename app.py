
from flask import Flask, request, jsonify
from pyngrok import ngrok
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import re
import os
from datetime import datetime
import json
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

# === SETUP DIRECTORIES AND FILE PATHS ===
#USED FOR WINDOWS
#os.makedirs("model", exist_ok=True)
#os.makedirs("csv_files", exist_ok=True)

#ADDED FOR LINUX
# Base directories for Linux/Windows compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_DIR = os.path.join(BASE_DIR, "csv_files")
#ADDED FOR LINUX
# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

#USED FOR WINDOWS
#MODEL_PATH_MED = "model/treatment_model_med.pkl"
#MLB_PATH_MED = "model/label_binarizer_med.pkl"
#USED FOR WINDOWS
#MODEL_PATH_PATHO = "model/treatment_model_path.pkl"
#MLB_PATH_PATHO = "model/label_binarizer_path.pkl"
#USED FOR WINDOWS
#MODEL_PATH_RADIO = "model/treatment_model_radio.pkl"
#MLB_PATH_RADIO = "model/label_binarizer_radio.pkl"
#USED FOR WINDOWS
#VECTORIZER_PATH = "model/vectorizer.pkl"

#ADDED FOR LINUX
MODEL_PATH_MED = os.path.join(MODEL_DIR, "treatment_model_med.pkl")
MLB_PATH_MED = os.path.join(MODEL_DIR, "label_binarizer_med.pkl")
#ADDED FOR LINUX
MODEL_PATH_PATHO = os.path.join(MODEL_DIR, "treatment_model_path.pkl")
MLB_PATH_PATHO = os.path.join(MODEL_DIR, "label_binarizer_path.pkl")
#ADDED FOR LINUX
MODEL_PATH_RADIO = os.path.join(MODEL_DIR, "treatment_model_radio.pkl")
MLB_PATH_RADIO = os.path.join(MODEL_DIR, "label_binarizer_radio.pkl")
#ADDED FOR LINUX
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


#USED FOR WINDOWS
#TRAINING_CSV = "csv_files/training_data.csv"
#FREQ_CSV = "csv_files/label_frequencies.csv"   # NEW: frequency counters

#ADDED FOR LINUX
TRAINING_CSV = os.path.join(CSV_DIR, "training_data.csv")
FREQ_CSV = os.path.join(CSV_DIR, "label_frequencies.csv")





# === LOAD OR INITIALIZE ML MODELS ===
def load_or_create_model(model_path, mlb_path):
    try:
        model = joblib.load(model_path)
        mlb = joblib.load(mlb_path)
        print(f" Loaded existing model: {model_path}")
    except Exception:
        # create a OneVsRest wrapper for multi-label classification
        base = SGDClassifier(loss="log_loss", max_iter=1000)
        model = OneVsRestClassifier(base)
        mlb = MultiLabelBinarizer()
        print(f" Created new model: {model_path}")
    return model, mlb

model_med, mlb_med = load_or_create_model(MODEL_PATH_MED, MLB_PATH_MED)
model_patho, mlb_patho = load_or_create_model(MODEL_PATH_PATHO, MLB_PATH_PATHO)
model_radio, mlb_radio = load_or_create_model(MODEL_PATH_RADIO, MLB_PATH_RADIO)

# === LOAD OR CREATE VECTORIZER ===
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(" Loaded existing vectorizer.")
except:
    vectorizer = TfidfVectorizer()
    print(" Created new TfidfVectorizer.")

# === LOAD REFERENCE DATA ===
def load_csv_safe(path, encoding='utf-8'):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df
        except pd.errors.EmptyDataError:
            print(f" File {path} is empty. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

df_disease = load_csv_safe(os.path.join(CSV_DIR, "disease_treatments_directory.csv"), encoding='latin-1')
if not df_disease.empty:
    df_disease.columns = [c.lower().strip() for c in df_disease.columns]

# ensure vectorizer has a baseline vocabulary
if not os.path.exists(VECTORIZER_PATH):
    disease_names = df_disease['disease'].dropna().tolist() if 'disease' in df_disease.columns else []
    vectorizer.fit(disease_names if disease_names else ["dummy disease"])
    joblib.dump(vectorizer, VECTORIZER_PATH)

df_tests = load_csv_safe(os.path.join(CSV_DIR, "radiology_pathology_test_terms.csv"), encoding='latin-1')
if not df_tests.empty:
    df_tests.columns = [c.lower().strip() for c in df_tests.columns]

# === EXTRACT TEST TERMS ===
def expand_terms(series):
    all_terms = []
    if series is None:
        return []
    for cell in pd.Series(series).dropna():
        for term in str(cell).split(","):
            term = term.strip()
            if term:
                all_terms.append(term)
    return sorted(set(all_terms))

pathology_terms = expand_terms(df_tests.get("pathology", []))
radiology_terms = expand_terms(df_tests.get("radiology", []))
print(f"Loaded {len(pathology_terms)} pathology terms and {len(radiology_terms)} radiology terms.")

# === LOAD SPACY MODELS ===
nlp_med7 = spacy.load("en_core_med7_lg")
if "sentencizer" not in nlp_med7.pipe_names:
    nlp_med7.add_pipe("sentencizer")
nlp_general = spacy.load("en_core_web_sm")

# === CREATE PHRASEMATCHER ONCE ===
matcher = PhraseMatcher(nlp_general.vocab, attr="LOWER")
if pathology_terms:
    matcher.add("PATHOLOGY TEST", [nlp_general.make_doc(t) for t in pathology_terms])
if radiology_terms:
    matcher.add("RADIOLOGY TEST", [nlp_general.make_doc(t) for t in radiology_terms])

# === FLASK APP INIT ===
app = Flask(__name__)

# === HELPER FUNCTIONS ===
def clean_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def merge_medication_entities(doc):
    merged = []
    skip_next = False
    for i, ent in enumerate(doc.ents):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(doc.ents) and doc.ents[i + 1].start == ent.end:
            merged_text = ent.text + " " + doc.ents[i + 1].text
            merged.append((ent.label_, merged_text))
            skip_next = True
        else:
            merged.append((ent.label_, ent.text))
    return merged

def extract_medications(text):
    cleaned_text = clean_text(text)
    doc_med = nlp_med7(cleaned_text)
    ents = merge_medication_entities(doc_med)
    medications = []
    INVALID_DRUGS = {"generic"}
    current_med = None
    for label, ent_text in ents:
        if label == "DRUG":
            drug_name = ent_text.strip()
            if drug_name.lower() in INVALID_DRUGS:
                continue
            existing = next((m for m in medications if m.get("drug", "").lower() == drug_name.lower()), None)
            if existing:
                current_med = existing
            else:
                current_med = {"drug": drug_name}
                medications.append(current_med)
        elif label in ["STRENGTH", "FORM", "DOSAGE", "ROUTE", "FREQUENCY", "DURATION"]:
            if current_med is not None:
                current_med[label.lower()] = ent_text
    return medications if medications else []

def extract_tests(text):
    cleaned_text = clean_text(text)
    doc = nlp_general(cleaned_text)
    matches = matcher(doc)
    patho, radio = [], []
    for match_id, start, end in matches:
        label = nlp_general.vocab.strings[match_id]
        span_text = doc[start:end].text
        if label == "PATHOLOGY TEST":
            patho.append(span_text)
        elif label == "RADIOLOGY TEST":
            radio.append(span_text)
    return list(set(patho)) or [], list(set(radio)) or []

# predict helper: mode="med" returns med dicts, else returns label strings
def predict_safe(model, mlb, X_enc, mode="med"):
    try:
        pred = model.predict(X_enc)
        inv = mlb.inverse_transform(pred)
        if not inv or not inv[0]:
            return []
        if mode == "med":
            return [label_to_med(l) for l in inv[0]]
        return list(inv[0])
    except Exception:
        return []

# === Option B: Convert med dict <-> label string ===
ATTRIBUTES = ["drug", "strength", "form", "dosage", "route", "frequency", "duration"]

def med_to_label(med):
    parts = []
    for attr in ATTRIBUTES:
        value = med.get(attr, "")
        parts.append(f"{attr.capitalize()}:{value}")
    return "|".join(parts)

def label_to_med(label_str):
    med = {}
    for part in label_str.split("|"):
        if ":" in part:
            key, val = part.split(":", 1)
            med[key.lower()] = val if val else None
    return med

# === ROUTES ===
@app.route('/')
def home():
    return jsonify({
        "message": "Flask API is running successfully!",
        "endpoints": [
            "/extract  – POST {'disease_name': 'actual_disease_name'}",
            "/extract_get?disease_name=actual_disease_name",
            "/submit_feedback – POST doctor feedback"
        ]
    })

@app.route("/extract_get", methods=["GET"])
def extract_info_get():
    disease_name = request.args.get("disease_name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Please provide a disease_name using ?disease_name=<name>"}), 400
    row = df_disease[df_disease["disease"].str.lower() == disease_name] if not df_disease.empty else pd.DataFrame()
    if row.empty:
        return jsonify({"error": f"Disease '{disease_name}' not found"}), 404
    row = row.iloc[0]
    clinical_text = str(row.get("text", ""))
    medications = extract_medications(clinical_text)
    patho_tests, radio_tests = extract_tests(clinical_text)
    result = {
        "disease_name": disease_name,
        "medications": medications,
        "pathology_tests": patho_tests,
        "radiology_tests": radio_tests
    }
    return jsonify(result)

# ---------------------------------------------------------------------
# helper: update frequency CSV
# ---------------------------------------------------------------------
def update_label_frequencies(disease, label_type, labels):
    """
    label_type: 'medication'|'pathology'|'radiology'
    labels: list of strings (for meds we pass med labels (serialized) as strings)
    """
    if not labels:
        return
    # ensure file exists
    if os.path.exists(FREQ_CSV):
        df = pd.read_csv(FREQ_CSV)
    else:
        df = pd.DataFrame(columns=["disease", "label_type", "label", "count"])
    for label in labels:
        # store label as string
        mask = (df["disease"] == disease) & (df["label_type"] == label_type) & (df["label"] == label)
        if mask.any():
            df.loc[mask, "count"] = df.loc[mask, "count"] + 1
        else:
            df = pd.concat([df, pd.DataFrame([{"disease": disease, "label_type": label_type, "label": label, "count": 1}])], ignore_index=True)
    df.to_csv(FREQ_CSV, index=False)

# ---------------------------------------------------------------------
# submit_feedback: append record, update frequencies, retrain from accumulated feedback
# ---------------------------------------------------------------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    global model_med, mlb_med, model_patho, mlb_patho, model_radio, mlb_radio, vectorizer

    data = request.get_json()
    disease_name = data.get("disease_name")
    suggested = data.get("suggested", {})
    final = data.get("final", {})
    if not disease_name or suggested is None or final is None:
        return jsonify({"error": "Missing disease_name, suggested, or final data"}), 400

    # --- Convert meds to Option B labels ---
    final_med_labels = [med_to_label(m) for m in final.get("medications", [])]
    final_patho_labels = final.get("pathology_tests", [])
    final_radio_labels = final.get("radiology_tests", [])

    # --- Save feedback record to TRAINING_CSV ---
    record = {
        "disease_name": disease_name,
        "suggested": json.dumps(suggested),
        "final": json.dumps(final),
        "final_med_labels": json.dumps(final_med_labels),
        "final_patho_labels": json.dumps(final_patho_labels),
        "final_radio_labels": json.dumps(final_radio_labels),
        "timestamp": datetime.now().isoformat()
    }
    df_feedback = load_csv_safe(TRAINING_CSV)
    df_feedback = pd.concat([df_feedback, pd.DataFrame([record])], ignore_index=True)
    df_feedback.to_csv(TRAINING_CSV, index=False)

    # --- Update frequency counters (persisted) ---
    # For meds, we store serialized med labels (strings) so counting is straightforward
    update_label_frequencies(disease_name, "medication", final_med_labels)
    update_label_frequencies(disease_name, "pathology", final_patho_labels)
    update_label_frequencies(disease_name, "radiology", final_radio_labels)

    # --- Retrain models from the full feedback dataset (TRAINING_CSV) ---
    # This avoids partial-fit shape issues and ensures consistent mlb classes.
    try:
        df_feedback = load_csv_safe(TRAINING_CSV)
        if not df_feedback.empty:
            # Build X (disease_name) and y lists for each model
            X_all = df_feedback["disease_name"].fillna("").astype(str).tolist()

            # Vectorize (re-fit on all disease names + existing df_disease)
            base_diseases = df_disease['disease'].dropna().astype(str).tolist() if not df_disease.empty else []
            vectorizer.fit(list(dict.fromkeys(base_diseases + X_all)))  # unique preserve order
            X_enc_all = vectorizer.transform(X_all)

            # Pathology: load final_patho_labels column (saved as JSON string)
            all_pathos = []
            for v in df_feedback["final_patho_labels"].fillna("[]"):
                try:
                    all_pathos.append(json.loads(v))
                except Exception:
                    # fallback if value is already a list or malformed
                    if isinstance(v, list):
                        all_pathos.append(v)
                    else:
                        all_pathos.append([])
            mlb_patho = MultiLabelBinarizer()
            if any(all_pathos):
                y_patho = mlb_patho.fit_transform(all_pathos)
                model_patho = OneVsRestClassifier(SGDClassifier(loss="log_loss", max_iter=1000))
                if y_patho.shape[1] > 0:
                    model_patho.fit(X_enc_all, y_patho)

            # Radiology
            all_radios = []
            for v in df_feedback["final_radio_labels"].fillna("[]"):
                try:
                    all_radios.append(json.loads(v))
                except Exception:
                    if isinstance(v, list):
                        all_radios.append(v)
                    else:
                        all_radios.append([])
            mlb_radio = MultiLabelBinarizer()
            if any(all_radios):
                y_radio = mlb_radio.fit_transform(all_radios)
                model_radio = OneVsRestClassifier(SGDClassifier(loss="log_loss", max_iter=1000))
                if y_radio.shape[1] > 0:
                    model_radio.fit(X_enc_all, y_radio)

            # Medications
            all_meds = []
            for v in df_feedback["final_med_labels"].fillna("[]"):
                try:
                    all_meds.append(json.loads(v))
                except Exception:
                    if isinstance(v, list):
                        all_meds.append(v)
                    else:
                        all_meds.append([])
            mlb_med = MultiLabelBinarizer()
            if any(all_meds):
                # all_meds is list of lists of serialized med strings
                y_med = mlb_med.fit_transform(all_meds)
                model_med = OneVsRestClassifier(SGDClassifier(loss="log_loss", max_iter=1000))
                if y_med.shape[1] > 0:
                    model_med.fit(X_enc_all, y_med)

            # Save updated vectorizer and mlbs/models
            joblib.dump(vectorizer, VECTORIZER_PATH)
            joblib.dump(model_med, MODEL_PATH_MED)
            joblib.dump(mlb_med, MLB_PATH_MED)
            joblib.dump(model_patho, MODEL_PATH_PATHO)
            joblib.dump(mlb_patho, MLB_PATH_PATHO)
            joblib.dump(model_radio, MODEL_PATH_RADIO)
            joblib.dump(mlb_radio, MLB_PATH_RADIO)

    except Exception as e:
        print("Retrain error:", e)

    return jsonify({"message": "Feedback submitted and models updated successfully!"})

# ---------------------------------------------------------------------
# extract endpoint: uses ML model predictions blended with frequency counts
# ---------------------------------------------------------------------
@app.route("/extract", methods=["POST"])
def extract_info():
    data = request.get_json()
    disease_name = data.get("disease_name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Please provide a disease_name"}), 400

    #  Check if models are actually valid and trained
    use_model = is_model_valid()
    print(f" Using {'MODEL' if use_model else 'CSV'} mode for:", disease_name)

    suggested_medications, suggested_pathology, suggested_radiology = [], [], []

    if use_model:
        X_enc = vectorizer.transform([disease_name])
        suggested_medications = predict_safe(model_med, mlb_med, X_enc, mode="med")
        suggested_pathology = predict_safe(model_patho, mlb_patho, X_enc, mode="test")
        suggested_radiology = predict_safe(model_radio, mlb_radio, X_enc, mode="test")



    # CSV fallback from original disease metadata
    row = df_disease[df_disease["disease"].str.lower() == disease_name] if not df_disease.empty else pd.DataFrame()
    meds_csv = []
    patho_csv = []
    radio_csv = []
    if not row.empty:
        row = row.iloc[0]
        clinical_text = str(row.get("text", ""))
        meds_csv = extract_medications(clinical_text)
        p, r = extract_tests(clinical_text)
        patho_csv = p
        radio_csv = r

    result = {
        "disease_name": disease_name,
        "suggested_medications": suggested_medications or meds_csv,
        "suggested_pathology": suggested_pathology or patho_csv,
        "suggested_radiology": suggested_radiology or radio_csv
    }

    # Blend with frequency counts so popular labels show on top
    if os.path.exists(FREQ_CSV):
        df_freq = pd.read_csv(FREQ_CSV)
        # helper to merge preserving freq order
        def merge_with_freq(disease, label_type, current_list):
            sub = df_freq[(df_freq["disease"] == disease) & (df_freq["label_type"] == label_type)]
            if sub.empty:
                return current_list
            # sort by frequency desc
            ordered = sub.sort_values("count", ascending=False)["label"].tolist()
            # ensure unique and keep frequency-first order, but keep any model-suggested labels appended if not present
            combined = []
            for lbl in ordered + current_list:
                if lbl not in combined:
                    combined.append(lbl)
            return combined

        # For medications we stored serialized med strings — convert predicted med dicts to serialized strings for merging,
        # then convert back to dicts for the result.
        # Build serialized list from predicted meds
        predicted_med_serial = []
        for med in result["suggested_medications"]:
            # if med is a dict (from label_to_med) convert back to serialized string
            if isinstance(med, dict):
                predicted_med_serial.append(med_to_label(med))
            elif isinstance(med, str):
                predicted_med_serial.append(med)
        freq_med_serial = merge_with_freq(disease_name, "medication", predicted_med_serial)

        # convert back to med dicts for final output (if possible)
        merged_meds_out = [label_to_med(s) if isinstance(s, str) and ":" in s else s for s in freq_med_serial]
        result["suggested_medications"] = merged_meds_out

        result["suggested_pathology"] = merge_with_freq(disease_name, "pathology", result["suggested_pathology"])
        result["suggested_radiology"] = merge_with_freq(disease_name, "radiology", result["suggested_radiology"])

    return jsonify(result)


def is_model_valid():
    """Check if models were actually trained (not just existing files)."""
    
    model_files = [
        #"model/treatment_model_med.pkl",
        MODEL_PATH_MED,
        #"model/treatment_model_path.pkl",
        MODEL_PATH_PATHO,
        #"model/treatment_model_radio.pkl",
        MODEL_PATH_RADIO,
        #"model/vectorizer.pkl"
        VECTORIZER_PATH
    ]
    # If model folder missing or empty -> invalid
    if not os.path.exists(MODEL_DIR):
        return False

    # If any model file missing or very small (untrained)
    for f in model_files:
        if not os.path.exists(f) or os.path.getsize(f) < 50000:  # ~50KB threshold
            return False

    # If no training data yet, also treat as unvalidated
    if not os.path.exists(TRAINING_CSV) or os.path.getsize(TRAINING_CSV) == 0:
        return False

    return True


# Optional debug endpoint to view frequency counts
@app.route("/view_frequencies", methods=["GET"])
def view_frequencies():
    if os.path.exists(FREQ_CSV):
        df = pd.read_csv(FREQ_CSV)
        return df.to_dict(orient="records")
    return jsonify([])


# === NGROK TUNNEL FOR TESTING ===
def start_ngrok():
    url = ngrok.connect(5000)
    print(f" Ngrok tunnel started: {url}")

if __name__ == "__main__":
    start_ngrok()
    app.run(port=5000)
