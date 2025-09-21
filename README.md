# Nexus

### Project Overview
Nexus is a hybrid intelligence platform designed to automate and streamline the resume evaluation process. It was developed to solve common challenges faced by placement teams and recruiters, such as a high volume of applications, time-consuming manual reviews, and inconsistent judgments. Nexus provides an efficient, scalable, and consistent system for candidate shortlisting.
The platform is built on a two-stage funnel architecture that balances speed with deep contextual analysis, ensuring cost-effectiveness and high-quality results.

[Access Nexus](https://code4edtechnexus.streamlit.app/)


[Submission Presentation](https://drive.google.com/drive/u/0/folders/1OMRfNjBtfuVr_6cvzaEcLNDidRyPKBUt)

### Key Features
- Flexible Inputs: The system supports multiple input methods, including single resumes, a batch of multiple files, or a single ZIP archive.
- Intelligent Analysis: A hybrid model combines a quantitative engine (local and fast) with a qualitative engine powered by a large language model (LLM).
- Dynamic Control: A "Skill Matching Strictness" slider allows recruiters to dynamically adjust the weight of the AI's semantic analysis versus traditional keyword matching.
- Centralized Dashboard: All analyses are automatically saved to an SQLite database, creating a permanent talent pool. The interactive dashboard provides a complete overview of all evaluations.
- Actionable Insights: The system generates three distinct, professional email drafts for acceptance, rejection, or an "Under Review" update to assist recruiters.
- Scalable Architecture: A two-stage funnel ensures cost-efficiency by using the powerful Gemini API for deep-dive analysis on only the top candidates identified during the rapid screening stage.

### Technical Architecture
Nexus employs a modular architecture with distinct engines for quantitative and qualitative analysis.
Core Technology Stack : 
- Frontend & Application Framework: Streamlit 
- Backend & Data Management: Python, SQLite, Pandas 
- File Processing: PyMuPDF and python-docx 
- Machine Learning & AI: scikit-learn, Sentence-Transformers, and Google Gemini API 

### Quantitative Engine (Local & Fast)
This engine is designed for rapid screening and is powered by local, resource-efficient models.
- Keyword Match: It uses TF-IDF Vectorization to calculate a baseline score based on keyword frequency.
- Semantic Match: It uses Sentence-Transformers to generate vector embeddings and calculate Cosine Similarity for a deep contextual understanding.

### Qualitative Engine (AI-Powered)
This engine provides rich, structured insights using a large language model.
- LLM Integration: It leverages the Google Gemini 1.5 Flash model for its balance of performance and a generous free-tier quota.
- Efficiency: The key innovation is a single, consolidated API call that requests a full analysis in one structured JSON object, which dramatically optimizes token usage and efficiency.
