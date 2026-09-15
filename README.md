# Semantic Resume Shortlisting App

This upgraded app compares one job description with multiple resumes using a
pretrained transformer, exact terminology, normalized skills, experience, and
education evidence.

It also includes automatic OCR fallback for scanned PDFs, local account
registration/login, and per-user saved analysis history in SQLite. Résumé files
and extracted résumé text are not stored in the database.

## Windows setup

```powershell
py -3.10 -m venv resume_ai_env
.\resume_ai_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The transformer downloads on first use. Keep the internet connected and do not
interrupt that first run. Later runs reuse the cached model.

EasyOCR also downloads its English OCR weights on first use. Normal text-based
PDFs do not invoke OCR.

## Score composition

- 55% transformer semantic similarity
- 20% normalized required-skill coverage
- 10% exact TF-IDF similarity
- 10% experience requirement
- 5% education requirement

These weights are starting values and must be calibrated with recruiter-reviewed
examples before production use.

## Files

- `Resume_Shortlisting_AI_Development.ipynb`: editable end-to-end learning flow
- `advanced_engine.py`: production scoring logic
- `document_parser.py`: PDF, DOCX and TXT extraction
- `database.py`: users, secure password hashes, and saved result history
- `app.py`: Streamlit interface
