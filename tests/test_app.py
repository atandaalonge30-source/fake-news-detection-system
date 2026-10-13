from app import app


def test_home_page_loads():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Fake News Detection System" in response.data


def test_health_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.MODEL_PATH", tmp_path / "missing-model.joblib")

    response = app.test_client().get("/health")

    assert response.status_code == 503
    assert response.json == {"status": "model_missing"}


def test_prediction_form_explains_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.MODEL_PATH", tmp_path / "missing-model.joblib")

    response = app.test_client().post("/", data={"text": "A long enough news article for this test."})

    assert response.status_code == 200
    assert b"Trained model not found" in response.data


def test_prediction_form_displays_model_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    class FakeScores(list):
        def argmax(self):
            return max(range(len(self)), key=self.__getitem__)

    class FakeModel:
        classes_ = ["fake", "real"]

        def predict_proba(self, texts):
            return [FakeScores([0.2, 0.8])]

    monkeypatch.setattr("app.MODEL_PATH", model_path)
    monkeypatch.setattr("app.joblib.load", lambda _: FakeModel())

    response = app.test_client().post("/", data={"text": "A long enough news article for this test."})

    assert response.status_code == 200
    assert b"Prediction: REAL" in response.data
    assert b"80.0%" in response.data
