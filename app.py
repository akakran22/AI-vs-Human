
from flask import Flask, render_template, request, session
import cloudpickle
import numpy as np
from lime.lime_text import LimeTextExplainer
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

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

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=["Human", "AI"])

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
    
    # Make prediction
    try:
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        # Get LIME explanation
        exp = explainer.explain_instance(
            text, 
            model.predict_proba, 
            num_features=10
        )
        
        # Extract top features
        lime_features = exp.as_list()
        top_features = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Format features for display
        features_data = []
        for word, weight in top_features:
            features_data.append({
                'word': word,
                'weight': weight,
                'contribution': 'AI' if weight > 0 else 'Human',
                'abs_weight': abs(weight)
            })
        
        result = {
            'text': text,
            'prediction': 'AI-Generated' if prediction == 1 else 'Human-Written',
            'prediction_class': prediction,
            'confidence': float(max(probability)) * 100,
            'human_prob': float(probability[0]) * 100,
            'ai_prob': float(probability[1]) * 100,
            'model_name': model_name,
            'features': features_data
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
    
    # Split texts by newlines
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
    
    # Make predictions
    try:
        predictions = model.predict(text_list)
        probabilities = model.predict_proba(text_list)
        
        results = []
        for i, text in enumerate(text_list):
            results.append({
                'index': i + 1,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'full_text': text,
                'prediction': 'AI-Generated' if predictions[i] == 1 else 'Human-Written',
                'confidence': float(max(probabilities[i])) * 100,
                'human_prob': float(probabilities[i][0]) * 100,
                'ai_prob': float(probabilities[i][1]) * 100
            })
        
        # Calculate statistics
        ai_count = sum(1 for p in predictions if p == 1)
        human_count = len(predictions) - ai_count
        
        stats = {
            'total': len(text_list),
            'ai_count': ai_count,
            'human_count': human_count,
            'ai_percentage': (ai_count / len(text_list)) * 100,
            'human_percentage': (human_count / len(text_list)) * 100,
            'avg_confidence': np.mean([max(p) for p in probabilities]) * 100
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