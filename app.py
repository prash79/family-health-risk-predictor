from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # allows Replit frontend to talk to this API

# ── Load all models and scalers ──────────────────────────────────────────────
def load_model(name):
    with open(f'models/{name}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/{name}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

heart_model,        heart_scaler        = load_model('heart')
hypertension_model, hypertension_scaler = load_model('hypertension')
diabetes_model,     diabetes_scaler     = load_model('diabetes')
thyroid_model,      thyroid_scaler      = load_model('thyroid')
obesity_model,      obesity_scaler      = load_model('obesity')
kidney_model,       kidney_scaler       = load_model('kidney')
cancer_model,       cancer_scaler       = load_model('cancer')

# ── Helper ───────────────────────────────────────────────────────────────────
def predict_risk(model, scaler, features, threshold=0.5):
    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = model.predict_proba(arr_scaled)[0][1]
    risk_pct = round(prob * 100, 1)
    risk_level = "High" if prob >= threshold else "Low"
    return {"risk_percentage": risk_pct, "risk_level": risk_level}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    d = request.get_json()
    # columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach
    features = [d['age'], d['sex'], d['cp'], d['trestbps'],
                d['chol'], d['fbs'], d['restecg'], d['thalach']]
    return jsonify(predict_risk(heart_model, heart_scaler, features, threshold=0.5))


@app.route('/predict/hypertension', methods=['POST'])
def predict_hypertension():
    d = request.get_json()
    # columns: age, salt_intake, stress_score, sleep_duration, bmi,
    #          family_history, exercise_level, smoking_status
    features = [d['age'], d['salt_intake'], d['stress_score'],
                d['sleep_duration'], d['bmi'], d['family_history'],
                d['exercise_level'], d['smoking_status']]
    return jsonify(predict_risk(hypertension_model, hypertension_scaler, features, threshold=0.4))


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    d = request.get_json()
    # columns: pregnancies, glucose, blood_pressure, bmi,
    #          diabetes_pedigree, age
    features = [d['pregnancies'], d['glucose'], d['blood_pressure'],
                d['bmi'], d['diabetes_pedigree'], d['age']]
    return jsonify(predict_risk(diabetes_model, diabetes_scaler, features, threshold=0.4))


@app.route('/predict/thyroid', methods=['POST'])
def predict_thyroid():
    d = request.get_json()
    # columns: age, sex, on_thyroxine, query_on_thyroxine,
    #          on_antithyroid_medication, sick, pregnant,
    #          query_hypothyroid, query_hyperthyroid, lithium,
    #          goitre, tumor, hypopituitary, psych,
    #          TSH measured, TSH, TT4 measured, TT4,
    #          T4U measured, T4U, FTI measured, FTI
    features = [d['age'], d['sex'], d['on_thyroxine'], d['query_on_thyroxine'],
                d['on_antithyroid_medication'], d['sick'], d['pregnant'],
                d['query_hypothyroid'], d['query_hyperthyroid'], d['lithium'],
                d['goitre'], d['tumor'], d['hypopituitary'], d['psych'],
                d['tsh_measured'], d['tsh'], d['tt4_measured'], d['tt4'],
                d['t4u_measured'], d['t4u'], d['fti_measured'], d['fti']]
    return jsonify(predict_risk(thyroid_model, thyroid_scaler, features, threshold=0.5))


@app.route('/predict/obesity', methods=['POST'])
def predict_obesity():
    d = request.get_json()
    # columns: age, gender, height, weight, calc, favc, fcvc, ncp,
    #          scc, smoke, ch2o, family_history, faf, tue, caec,
    #          mtrans_automobile, mtrans_bike, mtrans_motorbike,
    #          mtrans_public_transportation, mtrans_walking
    features = [d['age'], d['gender'], d['height'], d['weight'],
                d['calc'], d['favc'], d['fcvc'], d['ncp'],
                d['scc'], d['smoke'], d['ch2o'], d['family_history'],
                d['faf'], d['tue'], d['caec'],
                d['mtrans_automobile'], d['mtrans_bike'],
                d['mtrans_motorbike'], d['mtrans_public_transportation'],
                d['mtrans_walking']]
    return jsonify(predict_risk(obesity_model, obesity_scaler, features, threshold=0.5))


@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    d = request.get_json()
    # columns: age, bp, pc, pcc, ba, htn, dm, cad, appet, pe, ane
    features = [d['age'], d['bp'], d['pc'], d['pcc'], d['ba'],
                d['htn'], d['dm'], d['cad'], d['appet'], d['pe'], d['ane']]
    return jsonify(predict_risk(kidney_model, kidney_scaler, features, threshold=0.5))


@app.route('/predict/cancer', methods=['POST'])
def predict_cancer():
    d = request.get_json()
    # columns: gender, age, smoking, yellow_fingers, anxiety,
    #          peer_pressure, chronic_disease, fatigue, allergy,
    #          wheezing, alcohol, coughing, shortness_of_breath,
    #          swallowing_difficulty, chest_pain
    features = [d['gender'], d['age'], d['smoking'], d['yellow_fingers'],
                d['anxiety'], d['peer_pressure'], d['chronic_disease'],
                d['fatigue'], d['allergy'], d['wheezing'], d['alcohol'],
                d['coughing'], d['shortness_of_breath'],
                d['swallowing_difficulty'], d['chest_pain']]
    return jsonify(predict_risk(cancer_model, cancer_scaler, features, threshold=0.5))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)