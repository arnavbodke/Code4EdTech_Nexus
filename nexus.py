import streamlit as st
import pandas as pd
import fitz
import docx
import zipfile
import io
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import database

st.set_page_config(page_title="Nexus", layout="wide")
database.init_db()

load_dotenv()
try:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    GEMINI_API_CONFIGURED = True
except Exception as e:
    st.error("Failed To Configure Gemini API. Please Check Your API Key.")
    GEMINI_API_CONFIGURED = False

@st.cache_resource
def load_models():
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return st_model

st_model = load_models()

def get_gemini_response(prompt, model_name="gemini-1.5-flash-latest"):
    if not GEMINI_API_CONFIGURED: return "Error: Gemini API Is Not Configured."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "Error: API Quota Exceeded. Please Wait A Moment And Try Again."
        return f"Error Calling Gemini API: {e}"

def extract_text_from_file(file_stream, filename):
    if filename.lower().endswith('.pdf'):
        return "".join(page.get_text() for page in fitz.open(stream=file_stream, filetype="pdf"))
    elif filename.lower().endswith('.docx'):
        return "\n".join(para.text for para in docx.Document(file_stream).paragraphs)
    return ""

def analyze_resume(resume_text, jd_text, filename="N/A", strictness=60):
    semantic_weight = strictness / 100.0
    keyword_weight = 1.0 - semantic_weight
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    keyword_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    resume_embedding = st_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = st_model.encode(jd_text, convert_to_tensor=True)
    semantic_score = util.cos_sim(resume_embedding, jd_embedding).item()
    
    combined_score = (keyword_score * keyword_weight) + (semantic_score * semantic_weight)
    
    prompt = f"""
    Analyse the resume and job description. Return a single JSON object with keys: "profile", "jd_skills", "summary", and "interview_questions".

    1. **profile**: Extract 'name', 'email', 'phone', 'linkedin_url', 'github_url', 'skills' (list), and 'education' (list of objects with 'degree' and 'university'). Omit keys if not found.
    2. **jd_skills**: Extract skills from the job description as a list.
    3. **summary**: Provide a concise fit analysis as 3-4 markdown bullet points.
    4. **interview_questions**: Generate three insightful interview questions, each with the 'question' and a bulleted list under 'assesses:'.

    Resume Text: ```{resume_text}```
    Job Description: ```{jd_text}```
    """
    
    response_text = get_gemini_response(prompt)
    
    analysis_results = {
        'score': round(combined_score * 100, 2),
        'structured_data': {'name': filename.split('.')[0]},
        'filename': filename
    }

    try:
        cleaned_json = response_text.strip().replace("```json", "").replace("```", "")
        llm_results = json.loads(cleaned_json)
        
        analysis_results.update({
            'structured_data': llm_results.get('profile', analysis_results['structured_data']),
            'jd_skills': llm_results.get('jd_skills', []),
            'qualitative_summary': llm_results.get('summary', "No Summary Generated."),
            'interview_questions': llm_results.get('interview_questions', [])
        })
    except (json.JSONDecodeError, TypeError):
        analysis_results['qualitative_summary'] = "Error: Could Not Parse AI Analysis. The Quantitative Score Is Still Valid."

    return analysis_results

def display_results(results):
    profile = results.get('structured_data', {})
    score = results.get('score', 0)
    
    st.markdown("---")
    st.header("Analysis Report")
    
    tab1, tab2, tab3 = st.tabs(["**Overview & Score**", "**Skill Analysis**", "**Interview Preparation**"])
    with tab1:
        st.subheader(f"Profile: {profile.get('name', 'N/A')}")
        
        contact_info = []
        if 'email' in profile: contact_info.append(f"Email : {profile['email']}")
        if 'phone' in profile: contact_info.append(f"Phone : {profile['phone']}")
        if 'linkedin_url' in profile: contact_info.append(f"LinkedIn : [LinkedIn]({profile['linkedin_url']})")
        if 'github_url' in profile: contact_info.append(f"GitHub : [GitHub]({profile['github_url']})")
        if contact_info: st.markdown(" | ".join(contact_info))

        if 'education' in profile and profile.get('education'):
            edu_list = []
            for edu in profile['education']:
                if isinstance(edu, dict):
                    edu_list.append(f"{edu.get('degree', '')} From {edu.get('university', '')}")
                elif isinstance(edu, str): edu_list.append(edu)
            if edu_list: st.markdown(f"**Education:** {', '.join(edu_list)}")
            
        st.subheader("Relevance Score")
        st.metric(label="Overall Match", value=f"{score}%")
        st.progress(score / 100)
            
        st.subheader("Profile Summary and Feedback")
        summary = results.get('qualitative_summary', 'N/A')
        if isinstance(summary, list):
            st.markdown("\n".join(summary))
        else:
            st.markdown(summary)

    with tab2:
        st.subheader("Skill Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Skills Found In Resume**")
            resume_skills = profile.get('skills', [])
            if resume_skills and isinstance(resume_skills, list): st.markdown(", ".join(f"`{s}`" for s in resume_skills))
            else: st.warning("No Skills Extracted.")
        with col2:
            st.write("**Skills Required By Job**")
            jd_skills = results.get('jd_skills', [])
            if jd_skills and isinstance(jd_skills, list): st.markdown(", ".join(f"`{s}`" for s in jd_skills))
            else: st.warning("No Skills Extracted.")
    
    with tab3:
        st.subheader("Generated Interview Questions")
        interview_questions = results.get('interview_questions', [])
        if isinstance(interview_questions, list) and interview_questions:
            for i, q_data in enumerate(interview_questions):
                st.markdown(f"**{i+1}. {q_data.get('question', 'N/A')}**")
                assesses_list = q_data.get('assesses', [])
                if assesses_list:
                    formatted_assesses = "\n".join([f"   - {item}" for item in assesses_list])
                    st.markdown(f"   *Assesses:*\n{formatted_assesses}")
        elif isinstance(interview_questions, str):
             st.markdown(interview_questions)
        else:
            st.warning("No Interview Questions Were Generated.")

    with st.expander("**Connect With Candidate**"):
        email_key_suffix = profile.get('name', 'default_email')
        if f'email_draft_{email_key_suffix}' not in st.session_state:
            st.session_state[f'email_draft_{email_key_suffix}'] = ""

        summary = results.get('qualitative_summary', 'A Review Of Your Resume Has Been Completed.')
        candidate_name = profile.get('name', 'Candidate')
        
        b_col1, b_col2, b_col3 = st.columns(3)
        
        with b_col1:
            if st.button("Acceptance", use_container_width=True):
                with st.spinner("Drafting Acceptance Email"):
                    prompt = f"Write a professional acceptance email to {candidate_name}. State that we are pleased to offer them the position. Mention the next steps will be shared by the HR team shortly. Sign off as 'The Placement Team'."
                    st.session_state[f'email_draft_{email_key_suffix}'] = get_gemini_response(prompt)
        with b_col2:
            if st.button("Rejection", use_container_width=True):
                with st.spinner("Drafting Rejection Email"):
                    prompt = f"Write a polite and respectful rejection email to {candidate_name}. Thank them for their interest and the time they invested. Mention that the competition was strong and that their profile will be kept on file for future opportunities. Based on this AI summary, provide one brief, constructive feedback point: {summary}. Sign off as 'The Placement Team'."
                    st.session_state[f'email_draft_{email_key_suffix}'] = get_gemini_response(prompt)
        with b_col3:
            if st.button("Under Review", use_container_width=True):
                with st.spinner("Drafting Update Email"):
                    prompt = f"Write a brief, professional email to {candidate_name} to provide an update. Inform them that their application is still under consideration and that the team is actively reviewing profiles. Thank them for their patience. Sign off as 'The Placement Team'."
                    st.session_state[f'email_draft_{email_key_suffix}'] = get_gemini_response(prompt)

        if st.session_state[f'email_draft_{email_key_suffix}']:
            st.subheader("Email Draft")
            with st.container(border=True):
                st.markdown(st.session_state[f'email_draft_{email_key_suffix}'])

def render_analyzer_page():
    st.header("Resume Evaluation")
    st.sidebar.subheader("Evaluation Setting")
    strictness = st.sidebar.slider("Skill Matching Strictness", 0, 100, 60, help="Controls The Weight Of AI's Contextual Understanding Vs. Direct Keyword Matching.")

    mode = st.radio("Select Mode", ["Single Resume", "Batch Upload"], horizontal=True)
    
    with st.container(border=True):
        if mode == "Single Resume":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Candidate Resume")
                uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"], label_visibility="collapsed")
            with col2:
                st.subheader("Job Description")
                uploaded_jd = st.file_uploader("Upload JD File", type=["pdf", "docx"], label_visibility="collapsed")
                st.markdown("<p style='text-align: center; color: grey;'> OR </p>", unsafe_allow_html=True)
                jd_text_paste = st.text_area("Paste JD Here", height=225, label_visibility="collapsed")

            if st.button("Analyse Single Resume", type="primary"):
                if 'last_batch_results' in st.session_state: del st.session_state.last_batch_results
                    
                jd_text = ""
                if uploaded_jd:
                    jd_text = extract_text_from_file(io.BytesIO(uploaded_jd.getvalue()), uploaded_jd.name)
                elif jd_text_paste:
                    jd_text = jd_text_paste
                
                if uploaded_resume and jd_text and GEMINI_API_CONFIGURED:
                    with st.spinner("Performing Analysis"):
                        resume_text = extract_text_from_file(io.BytesIO(uploaded_resume.getvalue()), uploaded_resume.name)
                        results = analyze_resume(resume_text, jd_text, filename=uploaded_resume.name, strictness=strictness)
                        conn = database.create_connection()
                        database.insert_evaluation(conn, results)
                        conn.close()
                        st.session_state.last_analysis = results
                        st.success("Analysis Complete And Saved To Dashboard.")
                else: st.warning("Please Provide A Resume, A Job Description, And Ensure API Key Is Configured.")

        elif mode == "Batch Upload":
            with st.container():
                st.subheader("Upload Resumes")
                batch_mode = st.radio("Batch Input Method", ["Upload ZIP File", "Upload Multiple Files"], horizontal=True, label_visibility="collapsed")
                files_to_process = []
                if batch_mode == "Upload ZIP File":
                    uploaded_zip = st.file_uploader("Upload A ZIP File", type=["zip"])
                    if uploaded_zip:
                        with zipfile.ZipFile(uploaded_zip, 'r') as z:
                            for filename in z.namelist():
                                if filename.lower().endswith(('.pdf', '.docx')):
                                    files_to_process.append({'name': filename, 'content': io.BytesIO(z.read(filename))})
                else:
                    uploaded_files = st.file_uploader("Upload Multiple Resume Files", type=["pdf", "docx"], accept_multiple_files=True)
                    if uploaded_files:
                        for f in uploaded_files:
                            files_to_process.append({'name': f.name, 'content': f})
            
            with st.container():
                st.subheader("Job Description For Batch")
                uploaded_jd_batch = st.file_uploader("Upload JD File For Batch", type=["pdf", "docx"], label_visibility="collapsed")
                st.markdown("<p style='text-align: center; color: grey;'> OR </p>", unsafe_allow_html=True)
                jd_text_paste_batch = st.text_area("Paste The Single JD For All Resumes", height=225)
            
            if st.button("Analyse Batch", type="primary"):
                if 'last_analysis' in st.session_state: del st.session_state.last_analysis

                jd_text_batch = ""
                if uploaded_jd_batch:
                    jd_text_batch = extract_text_from_file(io.BytesIO(uploaded_jd_batch.getvalue()), uploaded_jd_batch.name)
                elif jd_text_paste_batch:
                    jd_text_batch = jd_text_paste_batch

                if files_to_process and jd_text_batch and GEMINI_API_CONFIGURED:
                    with st.spinner("Processing Batch"):
                        batch_results_list = []
                        progress_bar = st.progress(0, text="Starting Batch Analysis")
                        conn = database.create_connection()
                        for i, file_info in enumerate(files_to_process):
                            resume_text = extract_text_from_file(file_info['content'], file_info['name'])
                            if resume_text:
                                results = analyze_resume(resume_text, jd_text_batch, filename=file_info['name'], strictness=strictness)
                                database.insert_evaluation(conn, results)
                                batch_results_list.append(results)
                                time.sleep(0.5)
                            progress_bar.progress((i + 1) / len(files_to_process), text=f"Analyzing {file_info['name']}")
                        conn.close()
                    st.session_state.last_batch_results = batch_results_list
                    st.success(f"Successfully Analysed And Saved {len(files_to_process)} Resumes")
                else: st.warning("Please Upload Files And Provide A Job Description.")

    if 'last_analysis' in st.session_state and st.session_state.last_analysis:
        display_results(st.session_state.last_analysis)
    
    if 'last_batch_results' in st.session_state and st.session_state.last_batch_results:
        with st.expander("**View Batch Analysis Results**", expanded=True):
            batch_results = st.session_state.last_batch_results
            candidate_names = [res.get('structured_data', {}).get('name', res.get('filename', 'Unknown')) for res in batch_results]
            selected_candidate_name = st.selectbox("Select A Candidate To View Their Report:", candidate_names)
            selected_result = next((res for res in batch_results if res.get('structured_data', {}).get('name', res.get('filename', 'Unknown')) == selected_candidate_name), None)
            if selected_result:
                display_results(selected_result)

def render_dashboard_page():
    st.header("Recruiter Dashboard")
    conn = database.create_connection()
    try:
        df = pd.read_sql_query("SELECT id, timestamp, candidate_name, score, verdict FROM evaluations ORDER BY timestamp DESC", conn)
    finally: conn.close()

    st.subheader("All Evaluations")
   
    job_role = st.text_input("Filter by Job Role", "")
    min_score = st.slider("Minimum Score (%)", 0, 100, 0, help="Filter candidates by minimum score")
    location = st.text_input("Filter by Location", "") 

    if not df.empty:
        df_display = df.copy()
        df_display.rename(columns={'id': 'ID', 'timestamp': 'Date', 'candidate_name': 'Candidate', 'score': 'Score', 'verdict': 'Verdict'}, inplace=True)

        
        if job_role:
            df_display = df_display[df_display['Candidate'].str.contains(job_role, case=False, na=False)]
        if min_score > 0:
            df_display = df_display[df_display['Score'] >= min_score]
       

        st.dataframe(df_display[['Date', 'Candidate', 'Score', 'Verdict']], use_container_width=True)

        st.subheader("View Detailed Analysis From Database")
        candidate_options = [f"ID {row['ID']}: {row['Candidate']} ({row['Date']})" for index, row in df_display.iterrows()] 
        selected_candidate_str = st.selectbox("Select A Candidate To View Their Full Report:", candidate_options)

        if selected_candidate_str:
            selected_id = int(selected_candidate_str.split(':')[0].replace('ID ', ''))
            conn = database.create_connection()
            try:
                full_results_df = pd.read_sql_query(f"SELECT full_results_json FROM evaluations WHERE id = {selected_id}", conn)
                if not full_results_df.empty:
                    full_results_json = full_results_df.iloc[0]['full_results_json']
                    results_dict = json.loads(full_results_json)
                    display_results(results_dict)
            finally: conn.close()

        with st.expander("Dashboard Management"):
            col1, col2 = st.columns(2)
            with col1:
                csv = df_display.to_csv(index=False).encode('utf-8') 
                st.download_button("Download Data As CSV", csv, "resume_evaluations.csv", "text/csv", key='download-csv')
            with col2:
                if st.button("Clear All Dashboard Data"): st.session_state.confirm_delete = True
            
            if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
                st.warning("**This Action Cannot Be Undone.**")
                if st.button("Yes, Clear All Data", type="primary"):
                    conn = database.create_connection()
                    database.clear_all_evaluations(conn)
                    conn.close()
                    if 'last_analysis' in st.session_state: del st.session_state.last_analysis
                    if 'last_batch_results' in st.session_state: del st.session_state.last_batch_results
                    del st.session_state.confirm_delete
                    st.success("Dashboard Data Has Been Cleared.")
                    st.rerun()
    else:
        st.warning("No Evaluations Found.")

st.title("Nexus : Candidate Evaluation System")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To", ["Analyse", "Dashboard"], label_visibility="collapsed")

with st.expander("How It Works", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Upload")
        with st.container(border=True):
            st.markdown("Provide Resumes (Single, Multiple, Or ZIP) And A Job Description.")
            st.markdown("Adjust The **Skill Strictness** In The Sidebar To Fine - Tune The Analysis.")
    with col2:
        st.subheader("2. Analyse")
        with st.container(border=True):
            st.markdown("The System Calculates A Relevance Score And Uses A Single, Optimized AI Call To Generate A Full Report.")
            st.markdown("All Results Are Automatically Saved To The Database.")
    with col3:
        st.subheader("3. Review")
        with st.container(border=True):
            st.markdown("Review The Detailed, Tabbed Report For Each Candidate On The **Analyse** Page.")
            st.markdown("Access, Filter, And Download All Past Analyses On The **Dashboard** Page.")

st.sidebar.info("This File Uses An API With A Limited Quota. If You Encounter Errors, The Daily Limit May Have Been Reached.")

if page == "Analyse":
    render_analyzer_page()
else:
    render_dashboard_page()
