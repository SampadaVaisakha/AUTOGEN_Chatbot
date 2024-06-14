import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'

# Load environment variables
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API key is missing or invalid. Please set the GEMINI_API_KEY environment variable.")

MAX_USER_REPLIES = 2
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 41,
        "config_list": [
            {
                "model": "gemini-1.5-pro-latest",
                "api_key": api_key,
                "api_type": "google"
            }
        ],
        "seed": 42
    },
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=MAX_USER_REPLIES,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    },
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def initiate_chat_with_retry(user_proxy, assistant, message, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            chat_response = user_proxy.initiate_chat(assistant, message=message, max_tokens=50)
            return chat_response
        except RuntimeError as e:
            if "429" in str(e):
                print(f"Quota exceeded. Retrying in 60 seconds... (Attempt {retries + 1}/{max_retries})")
                time.sleep(60)
                retries += 1
            else:
                raise e
    raise RuntimeError("Maximum retries exceeded. Quota issue persists.")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return "No file part"
        file = request.files['pdf']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            pdf_text = extract_text_from_pdf(filepath)

            question = request.form.get('question')
            if not question:
                return "No question provided"

            input_message = f"{question}\n\nPDF Content:\n{pdf_text}"
            chat_response = initiate_chat_with_retry(user_proxy, assistant, input_message)
            convs = chat_response.chat_history

            summary = None
            for conv in convs:
                if conv['role'] == 'assistant':
                    summary = conv['content']
                    break

            # Shorten summary to a specific length, e.g., first 200 characters
            if summary:
                short_summary = summary[:1500] + '...' if len(summary) > 1500 else summary
            else:
                short_summary = "No summary available"

            return render_template('index.html', short_summary=short_summary)

    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
