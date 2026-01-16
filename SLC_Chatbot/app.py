import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import LangChain components
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Modern LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

from langchain_classic.chains import RetrievalQA




# 1. Load the .env file
load_dotenv()

# -------------------- SETTINGS --------------------
DATA_FILE = "SLC Full Course Tracker Sheet.xls"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SLC Assistant", layout="centered")
st.title("South London College Chatbot")

# -------------------- DATA + VECTOR STORE --------------------
@st.cache_resource
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"File '{DATA_FILE}' not found in the folder. Please make sure the name matches exactly.")
        return None

    try:
        # We try to read it. If it's a modern file renamed .xls, openpyxl will handle it.
        # If it's an old .xls, you may need: pip install xlrd
        df = pd.read_excel(DATA_FILE) 
        df = df.fillna("N/A")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Build searchable text
        # Note: Ensure these column names match your Excel sheet EXACTLY
        df["combined"] = df.apply(
            lambda r: (
                f"Course: {r.get('Course Name', 'N/A')} | "
                f"Category: {r.get('Main Categories', 'N/A')} | "
                f"Price: {r.get('Standard Sale Price', 'N/A')} | "
                f"URL: {r.get('Course URL', 'N/A')}"
            ),
            axis=1
        )

        loader = DataFrameLoader(df, page_content_column="combined")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        return FAISS.from_documents(loader.load(), embeddings)

    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

# Load the database
db = load_data()

# -------------------- CHAT HISTORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- CHAT INPUT --------------------
user_query = st.chat_input("Ask me about South London College courses")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if db:
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key is missing. Check your .env file.")
        else:
            with st.chat_message("assistant"):
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=OPENAI_API_KEY
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 4})
                )

                with st.spinner("Thinking..."):
                    response = qa.invoke(user_query)
                    answer = response["result"]

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
    else:
        st.error("Database not initialized. Please check the Excel file and API key.")