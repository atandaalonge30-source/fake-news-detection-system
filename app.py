"""Small Flask demonstration interface. Run only after training a model."""
from __future__ import annotations
import os
from pathlib import Path
import joblib
from flask import Flask, jsonify, render_template_string, request

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_DIR / "artifacts" / "fake_news_model.joblib"))
app = Flask(__name__)

PAGE = """<!doctype html><html><head><title>Fake News Detector</title>
<style>body{font-family:Arial;max-width:800px;margin:40px auto;color:#19212b}textarea{width:100%;height:180px;padding:12px;font-size:16px}button{margin-top:14px;background:#146c94;color:white;border:0;padding:11px 18px;border-radius:5px} .result{margin-top:22px;padding:18px;background:#edf6fa;border-left:5px solid #146c94}</style></head>
<body><h1>Fake News Detection System</h1><p>Paste a news headline or article. This is a screening model, not a fact-checking authority.</p>
<form method='post'><textarea name='text' minlength='20' required placeholder='Paste news text here...'>{{ text }}</textarea><br><button>Analyse text</button></form>
{% if result %}<div class='result'><strong>Prediction: {{ result.label|upper }}</strong><br>Model confidence: {{ result.confidence }}<br><small>Always verify important claims with credible sources.</small></div>{% endif %}
{% if error %}<p style='color:#a11'>{{ error }}</p>{% endif %}</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    text, result, error = "", None, None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not MODEL_PATH.exists():
            error = (
                "Trained model not found. Add your labelled dataset, train the model "
                "with the commands in README.md, and deploy artifacts/fake_news_model.joblib."
            )
        elif len(text) < 20:
            error = "Please enter at least 20 characters."
        else:
            model = joblib.load(MODEL_PATH)
            scores = model.predict_proba([text])[0]
            position = scores.argmax()
            result = {"label": model.classes_[position], "confidence": f"{scores[position]:.1%}"}
    return render_template_string(PAGE, text=text, result=result, error=error)


@app.get("/health")
def health():
    if not MODEL_PATH.is_file():
        return jsonify(status="model_missing"), 503
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
