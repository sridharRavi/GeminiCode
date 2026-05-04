# assistant.py

import os
import pickle
import requests
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain.agents import AgentType, initialize_agent
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"  # Declare as a field so Pydantic accepts it

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "gemini"

# Initialize the LLM
llm = GeminiLLM()


# Load FAISS index and metadata
with open("faiss_store/data.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
metadata = data["metadata"]
model_name = data["model_name"]

# Load embedding model
model = SentenceTransformer(model_name)

# Load FAISS index
index = faiss.read_index("faiss_store/index.faiss")

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Function to query the assistant
def query_assistant(question, top_k=3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    
    context = "\n\n".join([texts[i] for i in I[0]])
    
    prompt = f"""You are an AI code assistant. Answer the question based on the code snippets below.

--- Code Snippets ---
{context}
----------------------

Question: {question}
Answer:"""
    
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text.strip()

#an agent to explain any function in the code base files 
def explain_function(function_name, top_k=3):
    question = f"What does the function `{function_name}` do?"
    return query_assistant(question, top_k)

##an agent to refactor/rewrite any function in the code base files 
def refactor_code(function_name, top_k=3):
    embedding = model.encode([function_name])
    D, I = index.search(np.array(embedding), top_k)

    for i in I[0]:
        chunk = texts[i]
        if f"def {function_name}" in chunk:
            prompt = (
                "You are a Python expert.\n\n"
                "Refactor the following function to improve clarity, structure, or performance "
                "Try reducing the runtimes from O(n^2) to lower runtimes"
                "without changing its behavior.\n\n"
                "```python\n"
                f"{chunk}\n"
                "```\n\n"
                "Refactored version:"
            )
            try:
                response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
                if hasattr(response, "text"):
                    return response.text.strip()
                else:
                    return "Gemini did not return a valid response."
            except Exception as e:
                return f"Gemini API error: {e}"

    return f"Could not find a function named `{function_name}` in the codebase."

##An agent to fetch python package related information from PyPI the API"
def fetch_pypi_info(package_name):
    """
    Fetch package information from the PyPI public API.
    Returns a short description and version of the package.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        summary = data["info"].get("summary", "No summary available.")
        version = data["info"].get("version", "Unknown version")
        return f"{package_name} (v{version}): {summary}"
    else:
        return f"Could not fetch info for '{package_name}' (status {response.status_code})"


def generate_tests(function_name, output_dir="generated_tests", top_k=3):
    # Retrieve the function using the same approach as refactor_code
    embedding = model.encode([function_name])
    D, I = index.search(np.array(embedding), top_k)

    for i in I[0]:
        chunk = texts[i]
        if f"def {function_name}" in chunk:
            prompt = (
                "Write Python unit tests for the following function using pytest. "
                "Ensure edge cases are covered, and import the function as if it's in the same directory.\n\n"
                f"```python\n{chunk}\n```\n\n"
                "Return only the test code."
            )
            try:
                response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
                test_code = response.text.strip()

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                test_file_path = os.path.join(output_dir, f"test_{function_name}.py")

                # Write the test file
                with open(test_file_path, "w") as f:
                    f.write(test_code)

                return f"Tests generated and saved to {test_file_path}"

            except Exception as e:
                return f"Gemini API error: {e}"

    return f"Could not find a function named `{function_name}` in the codebase."

def find_similar_code(query, top_k=3):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), top_k)

    results = []
    for i in I[0]:
        results.append(texts[i])

    return "\n\n---\n\n".join(results)

def trace_function_calls(function_name, top_k=5):
    query = f"Where is `{function_name}` used or called?"
    return query_assistant(query, top_k)

def find_bugs(function_name, top_k=3):
    embedding = model.encode([function_name])
    D, I = index.search(np.array(embedding), top_k)

    for i in I[0]:
        chunk = texts[i]
        if f"def {function_name}" in chunk:
            prompt = (
                "You are a senior Python engineer.\n\n"
                "Analyze the following function for bugs, edge cases, or bad practices.\n\n"
                f"```python\n{chunk}\n```\n\n"
                "List potential issues and improvements:"
            )

            response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
            return response.text.strip()

    return f"Function `{function_name}` not found."

def summarize_file(file_name, top_k=5):
    query = f"Summarize the purpose of file {file_name}"
    return query_assistant(query, top_k)


explain_function_tool = Tool(
    name = "ExplainFunction",
    func=explain_function,
    description="Explains the code function mentioned in the user query"
)

refactor_code_tool = Tool(
    name="RefactorCode",
    func=refactor_code,
    description="Refactors the code provided the function name"
)

fetch_pypi_info_tool = Tool(
    name="FetchPyPIInfoTool",
    func=fetch_pypi_info,
    description="Gets the package information provided the name"
)

generate_tests_tool = Tool(
    name="GenerateTestsTool",
    func=generate_tests,
    description="Write pyTest test files for the python files in the codebase"
)

find_similar_code_tool = Tool(
    name="FindSimilarCode",
    func=find_similar_code,
    description="Finds similar code snippets in the codebase based on a query"
)

trace_function_calls_tool = Tool(
    name="TraceFunctionCall",
    func=trace_function_calls,
    description="trace to the point where the functio is called"
)

find_bugs_tool = Tool(
    name="FindBugs",
    func=find_bugs,
    description="Find bugs in the code base and print them"
)

summarize_file_tool = Tool(
    name="SummarizeFile",
    func=summarize_file,
    description="Summarize the file structure and architecture present in the codebase"
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[explain_function_tool, refactor_code_tool, fetch_pypi_info_tool, generate_tests_tool,
           find_similar_code_tool, find_bugs_tool, summarize_file_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)


# Driver main function
if __name__ == "__main__":
    print("Code Assistant ready.")
    print("Commands:")
    print(" - explain <function_name>") #explains the function in your codebase
    print(" - refactor <function_name>")
    print(" - tests <file_name>  ")
    print(" - pypy <package_name> ")
    print("bugs in codebase - find bugs in your files")
    print("Trace function calls using trace <function_name>")
    print(" - Or ask a general question about the codebase.")
    print(" - trace <function_name>")
    print(" - summary <file_name>")
    print(" - Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            elif user_input.startswith("explain "):
                func_name = user_input[len("explain "):].strip()

                print("\n" + explain_function(func_name) + "\n")
            elif user_input.startswith("refactor "):
                func_name = user_input[len("refactor "):].strip()
                print("\n" + refactor_code(func_name) + "\n")
            elif user_input.startswith("pypi "):
                package = user_input[len("pypi "):].strip()
                print("\n" + fetch_pypi_info(package) + "\n")
            elif user_input.startswith("tests "):
                func_name = user_input[len("tests "):].strip()
                print("\n" + generate_tests(func_name) + "\n")
            elif user_input.startswith("similar "):
                query = user_input[len("similar "):].strip()
                print("\n" + find_similar_code(query) + "\n")
            elif user_input.startswith("bugs "):
                func_name = user_input[len("bugs "):].strip()
                print("\n" + find_bugs(func_name) + "\n")
            elif user_input.startswith("trace "):
                func_name = user_input[len("trace "):].strip()
                print("\n" + trace_function_calls(func_name) + "\n")
            elif user_input.startswith("summary "):
                file_name = user_input[len("summary "):].strip()
                print("\n" + summarize_file(file_name) + "\n")
            else:
                print("\n" + query_assistant(user_input) + "\n")
        except KeyboardInterrupt:
            break



'''
if __name__ == "__main__":
    print("Code Assistant ready.")
    print("Commands:")
    print(" - explain <function_name>") #explains the function in your codebase
    print(" - refactor <function_name>")
    print(" - tests <file_name>  ")
    print(" - pypy <package_name> ")
    print(" - Or ask a general question about the codebase.")
    print(" - Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            else:
                response = agent.invoke(user_input)
                print(response['output'])
        except KeyboardInterrupt:
            break


'''
