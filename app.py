import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pandas as pd
import re
from flask import Flask, request, jsonify

# Load Data
df = pd.read_csv('/content/Processed_booking_data.csv')

# Prepare Documents
documents = []
for ind in df.index:
    text = (
        f"Hotel: {df['hotel'][ind]} | "
        f"Lead Time: {df['lead_time'][ind]} | "
        f"Year: {df['arrival_date_year'][ind]} | "
        f"Month: {df['arrival_date_month'][ind]} | "
        f"Weekend Nights: {df['stays_in_weekend_nights'][ind]} | "
        f"Week Nights: {df['stays_in_week_nights'][ind]} | "
        f"Country: {df['country'][ind]} | "
        f"Reservation Status: {df['reservation_status'][ind]} | "
        f"Adults: {df['adults'][ind]} | "
        f"Meal: {df['meal'][ind]} | "
        f"Is Repeated Guest: {df['is_repeated_guest'][ind]} | "
        f"Reserved Room Type: {df['reserved_room_type'][ind]} | "
        f"Customer Type: {df['customer_type'][ind]} | "
        f"Total Revenue: {df['total_revenue'][ind]}\n#####\n"
    )
    doc = Document(page_content=text, metadata={"source": "local"})
    documents.append(doc)

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators="[#####]")
chunked_docs = text_splitter.split_documents(documents)

# Load Embeddings & Vector DB
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(chunked_docs, embeddings)

# Retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

text_generation_pipeline = transformers.pipeline(
    model=model, tokenizer=tokenizer, task="text-generation",
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1, return_full_text=False, max_new_tokens=300, temperature=0.3, do_sample=True
)

tiny_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
qa_chain = ConversationalRetrievalChain.from_llm(tiny_llm, retriever, return_source_documents=True)

# Flask App
app = Flask(__name__)

# Detect Calculation Query
def is_calculation_query(query):
    keywords = ["total", "sum", "average", "highest", "lowest", "count", "revenue", "price", "cancellations"]
    return any(word in query.lower() for word in keywords)

# Handle Calculation Query
def handle_calculation_query(query):
    if "total revenue" in query.lower():
        match = re.search(r"(\w+)\s+(\d{4})", query)
        if match:
            month, year = match.groups()
            filtered_df = df[(df['arrival_date_month'] == month) & (df['arrival_date_year'] == int(year))]
            total_revenue = filtered_df['total_revenue'].sum()
            return f"Total revenue for {month} {year} is {total_revenue:.2f}"
        return "Please specify a valid month and year."

    elif "room price per night of each hotel of each month" in query.lower():
        avg_price_per_hotel_month = df.groupby(['hotel', 'arrival_date_month'])['adr'].mean().reset_index()
        return avg_price_per_hotel_month.to_dict(orient="records")

    elif "cancellation rate" in query.lower() and "each hotel" in query.lower():
        cancellation_stats = df.groupby('hotel')['is_canceled'].agg(['sum', 'count']).reset_index()
        cancellation_stats['cancellation_rate'] = (cancellation_stats['sum'] / cancellation_stats['count']) * 100
        return cancellation_stats.to_dict(orient="records")

    elif "country wise guests" in query.lower():
        df['total_guests'] = df['adults'] + df.get('children', 0) + df.get('babies', 0)
        country_guests = df.groupby('country')['total_guests'].sum().sort_values(ascending=False)
        return country_guests.head(10).to_dict()

    return "I couldn't understand the calculation request."

# API Endpoint
@app.route('/query', methods=['POST'])
def query_model():
    data = request.json
    query = data.get("query", "")
    
    if is_calculation_query(query):
        response = handle_calculation_query(query)
    else:
        response = qa_chain.invoke({'question': query, 'chat_history': []})['answer']
    
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
