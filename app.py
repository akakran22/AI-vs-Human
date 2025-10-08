from flask import Flask, render_template, request
import cloudpickle
import numpy as np
from lime.lime_text import LimeTextExplainer
from markupsafe import Markup, escape
import secrets
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# Load models
try:
    with open('models/lightgbm_model.pkl', 'rb') as f:
        lgbm_model = cloudpickle.load(f)
    print("✓ LightGBM model loaded successfully")
except Exception as e:
    print(f"✗ Error loading LightGBM model: {e}")
    lgbm_model = None

try:
    with open('models/logreg_model.pkl', 'rb') as f:
        logreg_model = cloudpickle.load(f)
    print("✓ Logistic Regression model loaded successfully")
except Exception as e:
    print(f"✗ Error loading Logistic Regression model: {e}")
    logreg_model = None

# Initialize LIME explainer (order matches label semantics: 0=Human, 1=AI)
explainer = LimeTextExplainer(class_names=["Human", "AI"])

def get_class_positions(model):
    """
    Return (ai_pos, human_pos) as column indices for predict_proba.
    Works whether classes_ are ints (0/1) or strings ('Human'/'AI').
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        # Fallback to conventional binary order
        return 1, 0

    classes = np.array(classes)
    # String labels
    if classes.dtype.kind in {"U", "S", "O"}:
        classes_lower = np.array([str(c).lower() for c in classes])
        if "ai" in classes_lower:
            ai_pos = int(np.where(classes_lower == "ai")[0][0])
        else:
            ai_pos = 1 if classes_lower.size > 1 else 0
        if "human" in classes_lower:
            human_pos = int(np.where(classes_lower == "human")[0][0])
        else:
            human_pos = (1 - ai_pos) if classes_lower.size > 1 else 0
        return ai_pos, human_pos

    # Numeric labels
    ai_pos = int(np.where(classes == 1)[0][0]) if (classes == 1).any() else (1 if classes.size > 1 else 0)
    human_pos = int(np.where(classes == 0)[0][0]) if (classes == 0).any() else (1 - ai_pos if classes.size > 1 else 0)
    return ai_pos, human_pos

def is_ai_label(pred):
    """Return True if a predicted label corresponds to the AI class."""
    try:
        if isinstance(pred, (np.integer, int)):
            return int(pred) == 1
        return str(pred).lower() == "ai"
    except Exception:
        return False

def generate_highlighted_text(text, feature_weights):
    """Generate HTML with highlighted words based on LIME weights, safely escaped."""
    words = text.split()
    highlighted_words = []

    for word in words:
        clean_word = word.strip('.,!?;:"()[]{}').lower()
        weight = feature_weights.get(clean_word, 0.0)
        safe_word = escape(word)

        if weight != 0:
            intensity = min(abs(weight) * 5.0, 1.0)  # scale into [0,1]
            if weight > 0:  # supports AI
                color = f'rgba(255, 99, 132, {intensity * 0.4})'
                span = Markup(
                    f'<span style="background-color:{color};padding:2px 4px;border-radius:3px;font-weight:500;" '
                    f'title="AI: {weight:.4f}">{safe_word}</span>'
                )
            else:  # supports Human
                color = f'rgba(54, 162, 235, {intensity * 0.4})'
                span = Markup(
                    f'<span style="background-color:{color};padding:2px 4px;border-radius:3px;font-weight:500;" '
                    f'title="Human: {weight:.4f}">{safe_word}</span>'
                )
            highlighted_words.append(span)
        else:
            highlighted_words.append(safe_word)

    return Markup(' ').join(highlighted_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    model_choice = request.form.get('model', 'lgbm')

    if not text:
        return render_template('index.html', error="Please enter text to classify")

    # Select model
    if model_choice == 'lgbm' and lgbm_model:
        model = lgbm_model
        model_name = "LightGBM"
    elif model_choice == 'logreg' and logreg_model:
        model = logreg_model
        model_name = "Logistic Regression"
    else:
        return render_template('index.html', error="Selected model not available")

    try:
        # Class positions for probabilities
        ai_pos, human_pos = get_class_positions(model)

        # Prediction and probabilities
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        ai_prob = float(probability[ai_pos]) * 100.0
        human_prob = float(probability[human_pos]) * 100.0
        confidence = float(np.max(probability)) * 100.0

        # LIME explanation for the AI class column index
        exp = explainer.explain_instance(
            text,
            model.predict_proba,
            num_features=10,
            labels=[ai_pos]
        )
        lime_features_ai = exp.as_list(label=ai_pos)

        # Built-in LIME HTML for interactive view (robust tokenization/colors)
        lime_html = exp.as_html(labels=[ai_pos], predict_proba=True)

        # Extract top features by absolute weight
        top_features = sorted(lime_features_ai, key=lambda x: abs(x[1]), reverse=True)[:10]

        features_data = []
        for word, weight in top_features:
            features_data.append({
                'word': word,
                'weight': weight,
                'contribution': 'AI' if weight > 0 else 'Human',
                'abs_weight': abs(weight)
            })

        # Build mapping for highlighting (single-word tokens)
        feature_map = {w.strip('.,!?;:"()[]{}').lower(): wt for w, wt in lime_features_ai}
        highlighted_html = generate_highlighted_text(text, feature_map)

        result = {
            'text': text,
            'prediction': 'AI-Generated' if is_ai_label(prediction) else 'Human-Written',
            'prediction_class': int(prediction) if isinstance(prediction, (np.integer, int)) else str(prediction),
            'confidence': confidence,
            'human_prob': human_prob,
            'ai_prob': ai_prob,
            'model_name': model_name,
            'features': features_data,
            'highlighted_html': highlighted_html,
            'lime_html': Markup(lime_html),  # pass LIME HTML to template
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {str(e)}")

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    texts = request.form.get('texts', '').strip()
    model_choice = request.form.get('model', 'lgbm')

    if not texts:
        return render_template('batch.html', error="Please enter texts to classify")

    text_list = [t.strip() for t in texts.split('\n') if t.strip()]
    if not text_list:
        return render_template('batch.html', error="No valid texts found")

    # Select model
    if model_choice == 'lgbm' and lgbm_model:
        model = lgbm_model
        model_name = "LightGBM"
    elif model_choice == 'logreg' and logreg_model:
        model = logreg_model
        model_name = "Logistic Regression"
    else:
        return render_template('batch.html', error="Selected model not available")

    try:
        ai_pos, human_pos = get_class_positions(model)

        predictions = model.predict(text_list)
        probabilities = model.predict_proba(text_list)

        results = []
        for i, text in enumerate(text_list):
            probs = probabilities[i]
            ai_prob = float(probs[ai_pos]) * 100.0
            human_prob = float(probs[human_pos]) * 100.0
            prediction = predictions[i]
            results.append({
                'index': i + 1,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'full_text': text,
                'prediction': 'AI-Generated' if is_ai_label(prediction) else 'Human-Written',
                'confidence': float(np.max(probs)) * 100.0,
                'human_prob': human_prob,
                'ai_prob': ai_prob
            })

        ai_count = sum(1 for p in predictions if is_ai_label(p))
        human_count = len(predictions) - ai_count

        stats = {
            'total': len(text_list),
            'ai_count': ai_count,
            'human_count': human_count,
            'ai_percentage': (ai_count / len(text_list)) * 100.0,
            'human_percentage': (human_count / len(text_list)) * 100.0,
            'avg_confidence': float(np.mean([np.max(p) for p in probabilities])) * 100.0
        }

        return render_template('batch_result.html',
                               results=results,
                               stats=stats,
                               model_name=model_name)

    except Exception as e:
        return render_template('batch.html', error=f"Error during prediction: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
