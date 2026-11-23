import os
import sys
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.news_analyzer import NewsCirculationAnalyzer
from src.link_extractor import extract_content_from_url

# Pre-initialize analyzer at cold-start
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsCirculationAnalyzer()
    return _analyzer

SAMPLE_STORIES = [
    {
        "label": "Reuters Wire (Authentic)",
        "type": "real",
        "title": "Reuters: Fed holds interest rates steady amid moderated inflation",
        "text": "WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday and signaled borrowing costs are likely to stay on hold for an extended period as officials await more clarity on the trajectory of inflation."
    },
    {
        "label": "Viral Sensational Hoax (Fake)",
        "type": "fake",
        "title": "BOMBSHELL HOAX: Mindblowing secret plot exposed by whistleblowers",
        "text": "SHOCKING BOMBSHELL EXPOSED! Deep state whistleblowers leaked documents showing secret election rigging schemes and hidden alien technology suppressed by mainstream media! Unbelievable cover-up exposed!"
    },
    {
        "label": "Wikipedia Technology Reference (Authentic)",
        "type": "real",
        "title": "Natural Language Processing (Wikipedia excerpt)",
        "text": "Natural language processing is an interdisciplinary subfield of computer science and linguistics. It is primarily concerned with giving computers the ability to support and manipulate human speech."
    }
]

BENCHMARK_MODELS = [
    { "algorithm": "Linear SVM", "accuracy": "100.00%", "precision": "100.00%", "recall": "100.00%", "f1": "100.00%", "roc_auc": "100.00%", "train_time": "0.76 s", "status": "Deployed (Top Model)" },
    { "algorithm": "Random Forest", "accuracy": "99.95%", "precision": "100.00%", "recall": "99.90%", "f1": "99.95%", "roc_auc": "100.00%", "train_time": "1.79 s", "status": "Benchmarked" },
    { "algorithm": "Logistic Regression", "accuracy": "99.90%", "precision": "99.80%", "recall": "100.00%", "f1": "99.90%", "roc_auc": "100.00%", "train_time": "0.23 s", "status": "Benchmarked" },
    { "algorithm": "Gradient Boosting", "accuracy": "99.85%", "precision": "99.90%", "recall": "99.80%", "f1": "99.85%", "roc_auc": "99.90%", "train_time": "68.23 s", "status": "Benchmarked" },
    { "algorithm": "Multinomial Naive Bayes", "accuracy": "98.58%", "precision": "98.60%", "recall": "98.60%", "f1": "98.60%", "roc_auc": "99.94%", "train_time": "0.02 s", "status": "Benchmarked" }
]

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path in ('/api/health', '/api'):
            data = {
                "status": "healthy",
                "service": "VeriNews Fake News Detection API",
                "model": "Linear SVM (Top Benchmark Performer)",
                "dataset_samples": 9865,
                "author": "Yaswanth (92400120483) - Marwadi University"
            }
            self._respond_json(200, data)
        elif path == '/api/samples':
            self._respond_json(200, {"samples": SAMPLE_STORIES})
        elif path == '/api/benchmark':
            self._respond_json(200, {"benchmarks": BENCHMARK_MODELS})
        else:
            self._respond_json(404, {"error": f"Endpoint '{self.path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path not in ('/api/predict', '/api/analyze'):
            self._respond_json(404, {"error": f"Endpoint '{self.path}' not found."})
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._respond_json(400, {"error": "Empty request body."})
                return

            body = self.rfile.read(content_length)
            payload = json.loads(body.decode('utf-8'))

            url = payload.get("url", "").strip()
            text = payload.get("text", "").strip()
            source_metadata = None

            if url:
                try:
                    ext_data = extract_content_from_url(url)
                    text = ext_data.get("content", "")
                    source_metadata = {
                        "source_type": ext_data.get("source_type", "Extracted URL"),
                        "title": ext_data.get("title", ""),
                        "author": ext_data.get("author", ""),
                        "url": url,
                        "preview": ext_data.get("preview", "")
                    }
                except Exception as url_err:
                    self._respond_json(400, {"error": f"Failed to extract content from URL: {str(url_err)}"})
                    return

            if not text:
                self._respond_json(400, {"error": "No text or valid URL provided for analysis."})
                return

            analyzer = get_analyzer()
            analysis = analyzer.analyze_news(text, source_metadata=source_metadata)
            self._respond_json(200, analysis)

        except Exception as e:
            self._respond_json(500, {"error": f"Internal server error: {str(e)}"})

    def _respond_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.send_header('Content-Length', str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)
