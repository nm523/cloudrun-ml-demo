"""Gunicorn configuration."""

import json

bind = "0.0.0.0:8080"
"str: The host to listen on."

accesslog = "-"
"str: State where the logs should be sent. '-' is stdout."

access_log_format = json.dumps(
    {
        "remote_ip": "%(h)s",
        "request_id": "%({X-Request-Id}i)s",
        "response_code": "%(s)s",
        "request_method": "%(m)s",
        "request_path": "%(U)s",
        "request_querystring": "%(q)s",
        "request_timetaken": "%(D)s",
        "response_length": "%(B)s",
    }
)
"""str: A representation of the access log format in JSON so it can be queried programatically."""
