"""
app.py
Unified Application Launcher for Fake News Detection System.
DSE Capstone Project - Yaswanth (92400120483) - Marwadi University CSE.

Usage:
    python app.py            -> Launches the Interactive Web UI in your browser
    python app.py --cli      -> Runs in terminal CLI mode
    python app.py --text "..." -> Evaluates a single news text immediately
"""

import os
import sys
import argparse
import subprocess


def run_streamlit_ui():
    """Launches the Streamlit Web Application."""
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=false"]
    print("=" * 65)
    print("🚀 Launching VeriNews Web Application (Streamlit)...")
    print("The web UI will open automatically in your browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nWeb UI stopped.")


def run_cli_interactive():
    """Terminal interactive mode."""
    from src.predict import run_interactive_cli
    run_interactive_cli()


def run_single_prediction(text: str):
    """Predicts a single text sample."""
    from src.predict import FakeNewsClassifier
    classifier = FakeNewsClassifier()
    res = classifier.predict(text)
    
    print("\n" + "=" * 65)
    if res["prediction"] == "FAKE":
        status_line = f" [!] CLASSIFICATION: FAKE NEWS (Sensationalist / Fabricated) "
    else:
        status_line = f" [*] CLASSIFICATION: REAL NEWS (Credible / Authentic) "

    print(status_line.center(65, "="))
    print("=" * 65)
    print(f" Confidence Score : {res['confidence']:.2f}%")
    print(f" Key NLP Signals  : {', '.join(res['top_keywords'])}")
    print(f" Cleaned Input    : {res['cleaned_text'][:120]}...")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detection System - DSE Project (Yaswanth, 92400120483)"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run interactive mode in terminal instead of launching Web UI."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text (headline or article body) to classify directly in terminal."
    )
    args = parser.parse_args()

    if args.text:
        run_single_prediction(args.text)
    elif args.cli:
        run_cli_interactive()
    else:
        # Default is launching Web UI as requested by user
        run_streamlit_ui()


if __name__ == "__main__":
    main()
