from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
from flask import Flask, jsonify
import json
import random




app = Flask(__name__)

load_dotenv()
os.getenv("GOOGLE_API_KEY")
GoogleGenerativeAIEmbeddings(model="models/embedding-001")  

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files")
    raw_text = ""
    for file in uploaded_files:
        # Process each PDF file
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)  
    return jsonify({"message": "Files processed successfully."})



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input_chain(user_question):

    chat_chain = get_conversational_chain()

    new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    docs = new_db.similarity_search(user_question)

    response = chat_chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json['question']
    response = user_input_chain(user_question)
    return jsonify({"response": response})



@app.route('/get_question', methods=['GET'])
def get_question():
    with open(r'csqa_rs_train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    r = random.randint(0, 13250)
    json_str = json_list[r]
    result = json.loads(json_str)

    question = result["question"]["stem"]
    choices = []
    for i in range(len(result["question"]["choices"])):
        choices.append(result["question"]["choices"][i]['text'])

    ans = result["question"]["choices"][index[result["answerKey"]]]["text"]

    response = {
        "question": question,
        "choices": choices,
        "answer": ans
    }
    return jsonify(response)


def prompt_for_story():
    prompt = """
    Generate a story prompt based on the provided parameters.
    Parameters:
    - num_characters (int): The number of characters in the story.
    - synopsis (str): A brief overview of the story plot.
    - environments (list): A list of key environments or settings in the story.
    - start (str): The initial situation or setting of the story.
    - situations (list): A list of key plot situations or events in the story.
    - end (str): The resolution or conclusion of the story.
    - moral (str): The moral lesson or message conveyed by the story.

    "Write a story involving {num_characters} characters. Here's a brief synopsis: {synopsis}. The story is set in {environments}. It begins with {start}, followed by {situations}. Finally, it concludes with {end}. The story's moral is: '{moral}'."
    Returns:
    - prompt (str): A story keeping provided parameters in account.
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt, input_variables=["num_character","synopsis","environments","start","situations","end","moral"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain    

def generate_story(num_character,synopsis,environments,start,situations,end,moral):

    chat_chain = prompt_for_story()

    response = chat_chain.invoke({num_character,synopsis,environments,start,situations,end,moral}, return_only_outputs=True)
    return response["output_text"]



@app.route('/story_prompt', methods=['POST'])
def story_prompt():

    num_character = request.json['noc']
    synopsis = request.json['syn']
    environments = request.json['env']
    start = request.json['start']
    situations= request.json['situation']
    end = request.json['end']
    moral = request.json['moral']

    response = generate_story(num_character,synopsis,environments,start,situations,end,moral)
    return jsonify({"response": response})


def save(username, password, additional_data):

    user = {'username':username,'password':password,'data':additional_data}
    with open(r'users.txt', 'a') as file_write:
        file_write.write(user)


@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    additional_data = request.json['additional_data']
    
    if not username or not password:
        return jsonify({'message': 'Username and password are required.'}), 400


    save(username, password, additional_data)

    return jsonify({'message': additional_data})



if __name__ == '__main__':
    app.run(port=5000)



