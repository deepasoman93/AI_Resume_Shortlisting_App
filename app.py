import hashlib

import streamlit as st
from sentence_transformers import SentenceTransformer

from advanced_engine import MODEL_NAME, rank_resumes_advanced
from database import authenticate, create_user, initialize_database, list_analyses, load_analysis, save_analysis
from document_parser import extract_document_text


st.set_page_config(page_title="Semantic Resume Shortlisting", page_icon="🧠", layout="wide")
initialize_database()


@st.cache_resource(show_spinner="Loading the transformer model for the first time…")
def load_semantic_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading OCR for the first scanned document…")
def load_ocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def show_results(results):
    st.subheader("Ranked shortlist")
    st.dataframe(results[["Rank", "Candidate", "Match %", "Semantic %", "Skill coverage %"]], hide_index=True, use_container_width=True)
    for _, row in results.iterrows():
        with st.expander(f"#{row['Rank']} · {row['Candidate']} · {row['Match %']:.1f}%"):
            st.success(row["Strengths"])
            st.warning(row["Gaps"])
            st.write("**Detected experience:**", row["Experience"])
            st.write("**Semantic similarity:**", f"{row['Semantic %']:.1f}%")
            st.write("**Required-skill coverage:**", f"{row['Skill coverage %']:.1f}%")
    st.download_button("Download complete analysis", results.to_csv(index=False).encode("utf-8"), "semantic_resume_shortlist.csv", "text/csv")


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🧠 Semantic Resume Shortlisting")
    login_tab, register_tab = st.tabs(["Login", "Create account"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Login", type="primary")
        if login_clicked:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose username")
            new_password = st.text_input("Choose password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_clicked = st.form_submit_button("Create account")
        if register_clicked:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_username, new_password)
                (st.success if ok else st.error)(message)
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user['username']}**")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()
    st.divider()
    st.subheader("Saved analyses")
    history = list_analyses(st.session_state.user["id"])
    selected_history = st.selectbox(
        "Open previous result",
        options=[None] + [item["id"] for item in history],
        format_func=lambda value: "Select…" if value is None else next(item["job_title"] for item in history if item["id"] == value),
    )
    if selected_history and st.button("Load result"):
        st.session_state.loaded_results = load_analysis(st.session_state.user["id"], selected_history)

st.title("🧠 Semantic Resume Shortlisting")
st.write("Rank multiple résumés against one job description using transformer embeddings and structured evidence.")
job_title = st.text_input("Job title / reference", placeholder="Senior Data Analyst")
left, right = st.columns(2)
with left:
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Or paste job description", height=240)
with right:
    st.subheader("2. Candidate résumés")
    resume_files = st.file_uploader("Upload multiple résumés", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    st.caption("Scanned PDFs automatically use OCR. The first OCR run downloads its recognition model.")

if st.button("Analyze and rank résumés", type="primary", use_container_width=True):
    try:
        final_jd, jd_ocr = extract_document_text(jd_file, load_ocr_reader) if jd_file else (jd_text, False)
        if not final_jd.strip():
            raise ValueError("Add or upload a job description before analyzing.")
        if not resume_files:
            raise ValueError("Upload at least one résumé before analyzing.")
        seen_hashes, resumes, skipped, ocr_files = set(), [], [], []
        for file in resume_files:
            digest = hashlib.sha256(file.getvalue()).hexdigest()
            if digest in seen_hashes:
                skipped.append(f"{file.name} (duplicate)")
                continue
            seen_hashes.add(digest)
            text, used_ocr = extract_document_text(file, load_ocr_reader)
            if used_ocr:
                ocr_files.append(file.name)
            if not text.strip():
                skipped.append(f"{file.name} (no readable text after extraction)")
            else:
                resumes.append((file.name.rsplit(".", 1)[0], text))

        model = load_semantic_model()
        results = rank_resumes_advanced(final_jd, resumes, lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        analysis_id = save_analysis(st.session_state.user["id"], job_title, results)
        st.session_state.loaded_results = results
        if jd_ocr or ocr_files:
            st.info("OCR used for: " + ", ".join(([jd_file.name] if jd_ocr else []) + ocr_files))
        if skipped:
            st.warning("Skipped: " + "; ".join(skipped))
        st.success(f"Analysis saved with ID {analysis_id}.")
    except Exception as error:
        st.error(str(error))

if "loaded_results" in st.session_state and st.session_state.loaded_results is not None:
    show_results(st.session_state.loaded_results)

st.caption("Decision-support only. Human review is required before any hiring decision.")
