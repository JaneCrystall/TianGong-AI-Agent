import asyncio
import logging
import os
import sys
import cohere

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from src.modules.tools.common.search_pinecone import SearchPinecone

logging.basicConfig(
    filename="pineconeDocs.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    force=True,
)

search_pinecone = SearchPinecone()
llm_model = st.secrets["llm_model"]
langchain_verbose = str(st.secrets["langchain_verbose"])
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["COHERE_API_KEY"] = st.secrets["cohere_api_key"]
co = cohere.Client(os.environ["COHERE_API_KEY"])

k = 40
extend = 0

query = "Impact of inconsistent and uncertain material intensity data in building stock analysis"

pinecone_docs = search_pinecone.sync_similarity(query=query, top_k=k, extend=extend)
# pinecone_docs = search_pinecone.sync_mmr(query=query, top_k=k)

contents = [item["content"] for item in pinecone_docs]

response = co.rerank(
        model="rerank-english-v2.0",
        query=query,
        documents=contents,
        top_n=20,
    )
contents = [result.document["text"] for result in response.results]


logging.info("Query: %s", query)
for i, content in enumerate(contents, start=1):
    logging.info("Content %d: %s", i, content)
    logging.info("")


# def get_queries_func_calling_chain():
#     func_calling_json_schema = {
#         "title": "get_querys_and_filters_to_search_database",
#         "description": "Extract the queries and filters for database searching",
#         "type": "object",
#         "properties": {
#             "query": {
#                 "title": "Query",
#                 "description": "Multiple arguements extracted for a vector database semantic search from user's query, separate queries with a semicolon",
#                 "type": "string",
#             },
#             "length": {
#                 "title": "Request Text Length",
#                 "description": "The length of the introduction requested by the user, in words",
#                 "type": "string",
#             },
#             "created_at": {
#                 "title": "Date Filter",
#                 "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
#                 "type": "string",
#             },
#         },
#         "required": ["query"],
#     }

#     prompt_func_calling_msgs = [
#         SystemMessage(
#             content="You are a world class algorithm for extracting the all arguments and filters from user's query, for searching vector database. Give the user's key arguements, extract and list all the key arguments that need to be claimed for an introduction. Each arguement must be conslusive, speccific, independent, and complete. Each arguement must be structured to facilitate separate searches in a vector database. Make sure to answer in the correct structured format. For example, user's argurment is 'Building material stock (MS) is important for human wellbeing but has large resource/energy impacts. Managing building MS is crucial for sustainability and climate goals.', your answer should be 'Building material stock (MS) is important for human wellbeing.', 'Building material stock has large resource/energy impacts.', and 'Managing building MS is crucial for sustainability and climate goals.'",
#         ),
#         HumanMessage(content="The User Query:"),
#         HumanMessagePromptTemplate.from_template("{input}"),
#     ]

#     prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

#     llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

#     func_calling_chain = create_structured_output_chain(
#         output_schema=func_calling_json_schema,
#         llm=llm_func_calling,
#         prompt=prompt_func_calling,
#         verbose=langchain_verbose,
#     )

#     return func_calling_chain


# async def main(query: str):
#     queries_func_calling_chain = get_queries_func_calling_chain()
#     queries_func_calling_chain_output = queries_func_calling_chain.run(query)
#     outline_response = queries_func_calling_chain_output.get("query")
#     queries = outline_response.split("; ")



#     
#     pinecone_docs = await asyncio.gather(
#         *[
#             search_pinecone.async_similarity(query=query, top_k=k, extend=extend)
#             for query in queries
#         ]
#     )
#     pinecone_contents = search_pinecone.get_contentslist(pinecone_docs)
#     for query, content in zip(queries, pinecone_contents):
#         logging.info("Query: %s", query)
#         for i, chunks in enumerate(content, start=1):
#             logging.info("Content %d: %s", i, chunks)
#             logging.info("")



# query = """Please craft an introductory section that encapsulates following key arguments. It must be longer than 1000 words.

# 1. Building material stock (MS) is important for human wellbeing but has large resource/energy impacts. Managing building MS is crucial for sustainability and climate goals.

# 2. Quantifying building material stock through a bottom-up method using material intensity (MI) coefficients is common, but MI data is scarce, inconsistent, and uncertain.

# 3. Harmonized MI datasets are needed for comparison across regions, but it's unclear which building attributes most influence MI.

# 4. Existing statistical methods have limitations in analyzing influences on MI due to assumptions and use of averages.

# 5. Machine learning techniques like Random Forest could identify relative importance of variables affecting MI while reducing uncertainty."""

# asyncio.run(main(query=query))
