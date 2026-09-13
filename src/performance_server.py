"""Minimal HTTP endpoint to expose performance metrics."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from src.performance_tracker import get_performance_tracker

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics" or self.path == "/performance":
            try:
                tracker = get_performance_tracker()
                report = tracker.get_performance_report()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "report": report}, default=str).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): pass  # suppress logs

def start_metrics_server(port: int = 8006):
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
