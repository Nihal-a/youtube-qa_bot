from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai._common import GoogleGenerativeAIError


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def generate_pet_name(animal_type: str, animal_color: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["animal_type", "animal_color"],
        template=(
            "I have a {animal_type} pet with color of {animal_color} and I want a cool name for it. "
            "Suggest me five cool names for my pet."
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key="pet_name")
    response = chain.invoke({"animal_type": animal_type, "animal_color": animal_color})
    return response["pet_name"]


def langchain_agent(query: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    tools = [wikipedia_tool]

    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke({"input": query})
    return result["output"]


def vector_db_from_youtube(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    try:
        db = FAISS.from_documents(docs, embeddings)
        return db
    except GoogleGenerativeAIError as e:
        print(f"Error creating vector DB: {e}")
        return None


def get_response(query: str, db, k: int = 4) -> str:
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join(d.page_content for d in docs)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    return response.replace("\n", "")


def get_youtube_qa_response(video_url: str, query: str) -> str:
    db = vector_db_from_youtube(video_url)
    if db:
        return get_response(query, db)
    return "Could not process the video. This is likely due to Google API quota exhaustion or an invalid video URL."
