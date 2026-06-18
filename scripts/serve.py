"""CLI: launch the dashboard at http://localhost:8050.

python scripts/serve.py
"""

from battery_rul.dashboard.app import app

if __name__ == "__main__":
    app.run(debug=False, port=8050)
