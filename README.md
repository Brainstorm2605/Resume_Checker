# Resume ATS Score Checker

**Boost your resume's chances of passing through Applicant Tracking Systems (ATS)!**

This free and open-source tool analyzes your resume and provides an ATS score, along with detailed feedback powered by a Large Language Model (LLM) and framework like NLTK, langchain, Ollama .

## Key Features

- **Free and Open-Source:** No hidden costs or subscriptions.
- **ATS Scoring Mechanisms:**
    - **Keyword Matching:** Measures the overlap between your resume's keywords and those in the job description.
    - **Cosine Similarity:** Calculates the semantic similarity between your resume and the job description using advanced TF-IDF techniques. 
- **LinkedIn JD Direct Comparison:**  Directly compare your resume to a LinkedIn job description by pasting the URL.
- **LLM-Powered Feedback:**  Receive constructive and personalized feedback on your resume's content, structure, and relevance to the job description, generated by the powerful Ollama LLM.

## Getting Started

### 1. Prerequisites

   - **Python 3.8 or higher:** [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Install Ollama

   - Follow the instructions at [https://ollama.com/download/windows](https://ollama.com/download/windows) to download and install Ollama.

### 3. Start the LLM

   - Open your command prompt or terminal and run the following command:
     ```bash
     ollama run llama3
     ```

### 4. Set up a Virtual Environment (Recommended)

   - Create a virtual environment to manage project dependencies:
     ```bash
     python -m venv venv  # Create a virtual environment named 'venv'
     ```
   - Activate the virtual environment:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

### 5. Install Project Requirements

   - Install the necessary Python libraries:
     ```bash
     pip install -r requirnment.txt
     ```

### 6. Run the Code

   - Execute the main script:
     ```bash
     python llmats.py
     ```

### 7. Using the Gradio Interface

   - Access the interactive interface in your web browser (the address will be shown in the terminal). 
   - **Upload Resume:** Select your resume file (PDF or DOCX).
   - **Provide Job Description (Choose one):**
     - **Paste LinkedIn Job URL:**  Directly analyze a LinkedIn job posting.
     - **Upload Job Description PDF:** Upload a PDF file of the job description.
     - **Paste Job Description Text:**  Copy and paste the job description into the text box.
   - **Click "Check ATS Score".**

## Example Usage


# For direct JD comparison from LinkedIn, it should be from LinkedIn Job URL not from the Job Board URL 
![Should be link from this page](https://github.com/Brainstorm2605/Resume_Checker/blob/main/Screenshot%202024-05-18%20203518.png)
![not from this page](https://github.com/Brainstorm2605/Resume_Checker/blob/main/Screenshot%202024-05-18%20203552.png)
