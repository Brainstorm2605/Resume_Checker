import nltk
import gradio as gr
import docx2txt
import PyPDF2
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from linkedin_to_pdf import linkedin_to_pdf  # Import the function from your other file

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# --- LLM Setup ---
llm = Ollama(model="llama3")
system_template = """
You are a very experienced ATS (Application Tracking System) bot with a deep understanding named BOb the Resume builder.
You will review resumes with or without job descriptions.
You are an expert in resume evaluation and provide constructive feedback with dynamic evaluation.
You should also provide an improvement table, taking into account:
- Content (Medium priority)
- Keyword matching (High priority)
- Hard skills (High priority)
- Soft skills (High priority)
- Overall presentation (Low priority)
"""
feedback_template = """
Resume Feedback Report
Here is the resume you provided:
{resume_text}
And the job description:
{jd_section}

Create the Improvement Table in relevance to the resume and give the consideration and suggestion for each section strictly following 
the pattern as below and don't just out this guided pattern :
| Area          | Consideration                                                   | Status | Suggestions |
| ------------- | --------------------------------------------------------------- | ------ | ----------- |
| Content       | Measurable Results: At least 5 specific achievements or impact. |  Low   |             |
|               | Words to avoid: Negative phrases or clichés.                    |        |             |
| Keywords      | Hard Skills: Presence and frequency of hard skills.             |  High  |             |
|               | Soft Skills: Presence and frequency of soft skills.             |        |             |
| Presentation  | Education Match: Does the resume list a degree that matches the job requirements? |  High   |             |

Strengths:
List the strengths of the resume here.

Detailed Feedback:
Provide detailed feedback on the resume's content, structure, grammar, and relevance to the job description.

Suggestions:
Provide actionable suggestions for improvement, including specific keywords to include and skills to highlight.
"""


def get_llm_feedback(resume_text, jd_text=None):
    """Gets feedback on the resume from the Ollama LLM."""
    jd_section = jd_text if jd_text else "No job description provided."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", feedback_template)
    ])
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    response = chain.invoke({
        "resume_text": resume_text,
        "jd_section": jd_section,
    })
    return response['text']

# Resume Evaluation Functions
def load_resume(path):
    if path.lower().endswith('.docx'):
        return docx2txt.process(path)
    elif path.lower().endswith('.pdf'):
        with open(path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    else:
        raise ValueError("Unsupported file format. Please provide a .docx or .pdf file.")

def load_job_description(path):
    if path.lower().endswith('.pdf'):
        with open(path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf file.")

def preprocess_text(text):
    """Preprocesses the text by lowercasing, tokenizing, removing stop words, and lemmatizing."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def extract_keywords(text):
    """Extracts keywords from the preprocessed text."""
    return set(preprocess_text(text).split())

def calculate_cosine_similarity(resume, jd):
    """Calculates the cosine similarity between the resume and job description using TF-IDF."""
    vectorizer = TfidfVectorizer()
    text = [resume, jd]
    tfidf_matrix = vectorizer.fit_transform(text)
    return cosine_similarity(tfidf_matrix)[0][1]

def evaluate_resume(resume_path, job_description, job_description_file, jd_link):
    """Evaluates the resume, calculates the ATS score, and generates LLM feedback."""
    resume = load_resume(resume_path)
    pre_resume = preprocess_text(resume)

    # Priority: JD Link > JD File > JD Text
    if jd_link:
        jd_folder = r"D:\ats\jd" # Path to the JD folder
        if not os.path.exists(jd_folder):
            os.makedirs(jd_folder)  # Create the folder if it doesn't exist

        random_number = random.randint(1000, 9999)
        pdf_path = linkedin_to_pdf(jd_link, jd_folder, random_number)
        if pdf_path:
            job_description = load_job_description(pdf_path) 
    elif job_description_file:
        job_description = load_job_description(job_description_file.name)
    
    jd_available = bool(job_description.strip())
    if jd_available:
        pre_jd = preprocess_text(job_description)
        resume_keywords = extract_keywords(pre_resume)
        jd_keywords = extract_keywords(pre_jd)

        keyword_match_count = len(resume_keywords.intersection(jd_keywords))
        keyword_score = (keyword_match_count / len(jd_keywords)) * 45 if len(jd_keywords) > 0 else 0

        cosine_sim_score = calculate_cosine_similarity(pre_resume, pre_jd) * 55

        total_score = keyword_score + cosine_sim_score
        total_score = min(total_score, 100)

        llm_feedback = "vhvh"
        return total_score, f"ATS Score: {total_score:.2f}%\n\nLLM Feedback:\n{llm_feedback}"

    else:
        llm_feedback = get_llm_feedback(resume)
        return 0, f"LLM Feedback:\n{llm_feedback}"

# Gradio Interface
def main():
    def update_output(resume_path, job_description, job_description_file, jd_link):
        score, feedback = evaluate_resume(resume_path, job_description, job_description_file, jd_link)
        color = "red" if score <= 35 else "orange" if score <= 60 else "yellow" if score <= 80 else "green"
        progress_html = f"""
        <div style='width: 100%; background-color: lightgray;'>
            <div style='width: {score}%; background-color: {color}; text-align: center; color: white;'>
                {score:.2f}%
            </div>
        </div>
        """
        return progress_html, feedback

    def toggle_inputs(job_description, job_description_file, jd_link):
        if job_description.strip():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif job_description_file:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        elif jd_link:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    with gr.Blocks() as demo:
        gr.Markdown("## Resume ATS Score Checker")
        
        resume_input = gr.File(label="Upload Resume (PDF or DOCX)")
        
        with gr.Column(visible=True) as col_link:
            jd_link = gr.Textbox(label="LinkedIn Job URL (Optional)")

        with gr.Column(visible=True) as col_text:
            job_description_text = gr.Textbox(label="Job Description (Optional - Paste here if available or upload from below options)", lines=5)
        
        with gr.Column(visible=True) as col_file:
            job_description_file = gr.File(label="Upload Job Description (PDF)")
        
        
        job_description_text.change(toggle_inputs, inputs=[job_description_text, job_description_file, jd_link], outputs=[col_text, col_file, col_link])
        job_description_file.change(toggle_inputs, inputs=[job_description_text, job_description_file, jd_link], outputs=[col_text, col_file, col_link])
        jd_link.change(toggle_inputs, inputs=[job_description_text, job_description_file, jd_link], outputs=[col_text, col_file, col_link])

        output_meter = gr.HTML()
        output_text = gr.Textbox(label="ATS Feedback")

        button = gr.Button("Check ATS Score")
        button.click(update_output, inputs=[resume_input, job_description_text, job_description_file, jd_link], outputs=[output_meter, output_text])
        
        demo.launch()

if __name__ == "__main__":
    main()