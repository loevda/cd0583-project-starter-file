import json
import pytest


class TestScoringEndpoint:
    """GET /scoring must return the F1 score."""

    def test_returns_200(self, flask_client):
        resp = flask_client.get("/scoring")
        assert resp.status_code == 200

    def test_returns_numeric_score(self, flask_client):
        resp = flask_client.get("/scoring")
        data = resp.get_json()
        assert data is not None
        # Should contain a numeric F1 score
        if isinstance(data, dict):
            score = data.get("score") or data.get("f1") or data.get("f1_score")
            assert score is not None, f"No score field in response: {data}"
            assert 0.0 <= float(score) <= 1.0
        elif isinstance(data, (int, float)):
            assert 0.0 <= data <= 1.0


class TestSummaryStatsEndpoint:
    """GET /summarystats must return mean, median, mode per column."""

    def test_returns_200(self, flask_client):
        resp = flask_client.get("/summarystats")
        assert resp.status_code == 200

    def test_returns_stats_data(self, flask_client):
        resp = flask_client.get("/summarystats")
        data = resp.get_json()
        assert data is not None
        assert len(data) > 0, "Empty summary stats response"


class TestDiagnosticsEndpoint:
    """GET /diagnostics must return timing, dependencies, and NA percentages."""

    def test_returns_200(self, flask_client):
        resp = flask_client.get("/diagnostics")
        assert resp.status_code == 200

    def test_contains_timing(self, flask_client):
        resp = flask_client.get("/diagnostics")
        data = resp.get_json()
        assert data is not None
        data_str = str(data).lower()
        # Should reference timing/execution
        assert any(k in data_str for k in ["time", "timing", "execution", "second"]), (
            f"No timing info in diagnostics response: {data}"
        )

    def test_contains_dependency_info(self, flask_client):
        resp = flask_client.get("/diagnostics")
        data = resp.get_json()
        data_str = str(data).lower()
        assert any(k in data_str for k in ["package", "depend", "version", "outdated"]), (
            f"No dependency info in diagnostics response: {data}"
        )

    def test_contains_na_percentages(self, flask_client):
        resp = flask_client.get("/diagnostics")
        data = resp.get_json()
        data_str = str(data).lower()
        assert any(k in data_str for k in ["na", "missing", "null", "percent"]), (
            f"No NA/missing data info in diagnostics response: {data}"
        )


class TestPredictionEndpoint:
    """POST /prediction must accept data and return predictions."""

    def test_returns_200(self, flask_client):
        # Send a minimal payload
        payload = {
            "lastmonth_activity": [100, 200],
            "lastyear_activity": [1000, 2000],
            "number_of_employees": [10, 20],
        }
        resp = flask_client.post(
            "/prediction",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_returns_predictions(self, flask_client):
        payload = {
            "lastmonth_activity": [100, 200],
            "lastyear_activity": [1000, 2000],
            "number_of_employees": [10, 20],
        }
        resp = flask_client.post(
            "/prediction",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data is not None
        # Should contain predictions (list of 0/1)
        if isinstance(data, dict):
            preds = data.get("predictions") or data.get("prediction")
            assert preds is not None, f"No predictions field in: {data}"
        elif isinstance(data, list):
            preds = data
        assert len(preds) == 2
