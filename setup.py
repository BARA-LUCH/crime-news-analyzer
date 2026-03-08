#!/usr/bin/env python3
"""
setup.py — Auto-installer for AI Crime News Analyzer
Run this once before launching the app:  python setup.py
Works on Windows, macOS, and Linux.
"""

import subprocess
import sys
import platform
import shutil

OS = platform.system()  # "Windows", "Darwin", "Linux"

def run(cmd, check=True):
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def step(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

# ── Step 1: Python version check ─────────────────────────
step("1/5  Checking Python version")
major, minor = sys.version_info[:2]
if major < 3 or minor < 9:
    print(f"  ❌ Python 3.9+ required. You have {major}.{minor}")
    sys.exit(1)
print(f"  ✅ Python {major}.{minor} — OK")

# ── Step 2: pip packages ─────────────────────────────────
step("2/5  Installing Python packages")
run(f"{sys.executable} -m pip install --upgrade pip")
run(f"{sys.executable} -m pip install -r requirements.txt")
print("  ✅ Python packages installed")

# ── Step 3: spaCy model ──────────────────────────────────
step("3/5  Downloading spaCy English model")
run(f"{sys.executable} -m spacy download en_core_web_sm")
print("  ✅ spaCy model ready")

# ── Step 4: Tesseract OCR ────────────────────────────────
step("4/5  Checking Tesseract OCR")

tesseract_found = shutil.which("tesseract") is not None

if tesseract_found:
    print("  ✅ Tesseract already installed")
else:
    print("  ⚠️  Tesseract not found — attempting install...")

    if OS == "Linux":
        run("sudo apt-get update -qq")
        run("sudo apt-get install -y tesseract-ocr tesseract-ocr-heb tesseract-ocr-ara")

    elif OS == "Darwin":
        if shutil.which("brew"):
            run("brew install tesseract tesseract-lang")
        else:
            print("""
  ❌ Homebrew not found. Install Tesseract manually:
     1. Install Homebrew: https://brew.sh
     2. Run: brew install tesseract tesseract-lang
""")

    elif OS == "Windows":
        print("""
  ⚠️  On Windows, install Tesseract manually:
     1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
     2. Run the installer (choose Hebrew + Arabic language packs)
     3. Add Tesseract to your PATH
     4. Re-run this script
""")

    # Verify
    if shutil.which("tesseract"):
        print("  ✅ Tesseract installed successfully")
    else:
        print("  ⚠️  Tesseract install may have failed — OCR features won't work")
        print("      Text paste mode will still work fine without it.")

# ── Step 5: .env file ────────────────────────────────────
step("5/5  Setting up environment file")
import os
if not os.path.exists(".env"):
    if os.path.exists(".env.example"):
        import shutil as sh
        sh.copy(".env.example", ".env")
        print("  ✅ Created .env from .env.example")
        print("  📝 Open .env and add your OpenAI API key (optional)")
    else:
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("  ✅ Created .env file")
else:
    print("  ✅ .env already exists")

# ── Done ─────────────────────────────────────────────────
print(f"""
{'='*55}
  ✅ Setup complete!

  To launch the app:
     streamlit run app.py

  Then open: http://localhost:8501

  Notes:
  - OpenAI API key is OPTIONAL (app works without it)
  - First run downloads ~1GB of AI models (one time only)
  - Use 'Paste Text' mode to test without Tesseract
{'='*55}
""")
