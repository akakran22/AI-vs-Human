# AI vs Human Text Classifier - Flask Web Application

A production-ready Flask web application for classifying text as AI-generated or human-written using trained machine learning models with explainable AI features.

> **Note**: This repository contains only the Flask web application. For model training code and dataset, please visit the [AI vs Human Classification Models Repository](link-to-models-repo).

[Live Application Link](https://ai-vs-human-qqla.onrender.com)

## 📋 Overview

This web application provides an intuitive interface for detecting AI-generated content using pre-trained machine learning models. It features single text classification, batch processing, and LIME-based explanations for prediction transparency.

### Key Features

- 🤖 **Dual Model Support**: LightGBM (92% accuracy) and Logistic Regression (90% accuracy)
- 📊 **LIME Explanations**: Feature importance visualization for prediction transparency
- 🔄 **Batch Processing**: Classify multiple texts simultaneously with statistics
- 📈 **Confidence Scoring**: Real-time probability distributions and confidence metrics
- 🎨 **Responsive UI**: Clean interface built with Jinja2 templates (no JavaScript)
- ☁️ **Cloud Deployed**: Production-ready deployment on Render

## 🏗️ Project Structure

```
ai-text-classifier-flask/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore file
├── README.md                      # This file
│
├── models/                        # Pre-trained models (not in repo)
│   ├── lightgbm_model.pkl        # LightGBM model
│   └── logreg_model.pkl          # Logistic Regression model
│
└── templates/                     # HTML templates
    ├── base.html                 # Base template with navigation
    ├── index.html                # Single prediction page
    ├── result.html               # Single prediction results
    ├── batch.html                # Batch prediction input
    ├── batch_result.html         # Batch prediction results
    └── about.html                # Project information
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Pre-trained models (download separately)

### Setup Instructions

1. **Clone the repository**
```bash
git clone repo link
cd ai-text-classifier-flask
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Pre-trained Models**

Download the trained models from the [Models Repository](link-to-models-repo) and place them in the `models/` directory:
- `models/lightgbm_model.pkl`
- `models/logreg_model.pkl`

5. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📦 Dependencies

```
Flask==2.3.0
numpy==2.0.0
pandas==2.0.3
scikit-learn==1.5.2
lightgbm==4.5.0
lime==0.2.0.1
cloudpickle==3.0.0
Werkzeug==2.3.0
```

## 🎯 Usage

### Single Text Classification

1. Navigate to the home page
2. Enter or paste text in the textarea
3. Select a model (LightGBM or Logistic Regression)
4. Click "Classify Text"
5. View results with:
   - Prediction (AI-Generated or Human-Written)
   - Confidence score
   - Probability distribution
   - LIME feature importance (top 10 words)

### Batch Processing

1. Click "Batch Prediction" in the navigation
2. Enter multiple texts (one per line)
3. Select a model
4. Click "Classify All Texts"
5. View:
   - Summary statistics
   - Individual predictions
   - Confidence scores for each text

## 🧠 Model Information

### LightGBM (Recommended)
- **Accuracy**: 92%
- **F1-Score**: 0.92
- **True Positive Rate**: 93%
- **True Negative Rate**: 90%
- **False Positive Rate**: 10%

### Logistic Regression
- **Accuracy**: 90%
- **F1-Score**: 0.90
- **True Positive Rate**: 92%
- **True Negative Rate**: 88%
- **False Positive Rate**: 12%

> **Model Training Details**: Models were trained on 58,537 text samples (29,395 AI-generated, 29,142 human-written) using TF-IDF vectorization with unigram and bigram features. For training code and methodology, visit the [Models Repository](link-to-models-repo).

## 🔍 How It Works

1. **Text Preprocessing**: Input text is processed using CountVectorizer and TF-IDF transformation
2. **Feature Extraction**: Unigrams and bigrams are extracted with stop word removal
3. **Model Prediction**: Pre-trained model generates probability scores for both classes
4. **LIME Explanation**: Identifies top contributing words for the prediction
5. **Result Display**: Shows prediction, confidence, and feature importance

## 🌐 Deployment

### Render Deployment

This application is deployed on Render. To deploy your own instance:

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Add environment variables (if needed)
6. Upload model files to Render or use external storage


## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page (single prediction) |
| POST | `/predict` | Single text classification |
| GET | `/batch` | Batch prediction page |
| POST | `/batch_predict` | Batch text classification |
| GET | `/about` | Project information |

## 🔧 Configuration

### Change Port

Edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### Production Mode

For production deployment, disable debug mode:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🐛 Troubleshooting

### Model Loading Errors

**Issue**: `Error loading model: No module named 'numpy._core'`

**Solution**:
```bash
pip install --upgrade numpy==2.0.0 scikit-learn==1.5.2
```

## 🔒 Security Considerations

- Models are loaded once at startup for performance
- Input validation and sanitization implemented
- Session management with secure keys
- Rate limiting recommended for production
- HTTPS strongly recommended for deployment

## 📈 Performance Optimization

- Models are cached in memory
- Batch processing for multiple predictions
- Efficient TF-IDF vectorization
- Minimal dependencies for faster load times

## 📸 Screenshots

### Single Prediction
![Single Prediction](images/single prediction.png)

### Batch Processing
![Batch Processing](images/batch prediction.png)

### LIME Explanations
![LIME Explanations](images/LIME explanations.png)

---

⭐ If you find this project useful, please consider giving it a star!
