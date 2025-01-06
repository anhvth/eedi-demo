from typing import Annotated, List, TypedDict
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from eedi.client import SearchMisconceptionEngine
from eedi.data.common import *
import pandas as pd

# Constants
SYSTEM_PROMPT = """You are an AI math tutor with access to specialized tools for personalized instruction. Your primary goal is to help students master mathematical concepts through targeted practice and misconception correction.

Available Tools:
1. find_related_problems(misconception: str)
- Use this to fetch practice problems when:
- Student shows confusion about a concept
- Additional practice is needed
- Reviewing specific topics

2. find_related_misconceptions(text_problem, answer_text, correct_answer)
- Use this to analyze student responses and identify misconceptions
- Help diagnose learning gaps
- Guide remedial instruction

Teaching Protocol:
1. For New Topics:
- Assess current understanding
- Explain core concepts
- Use find_related_problems for targeted practice

2. When Student Makes Mistakes:
- Use find_related_misconceptions to diagnose issues
- Provide clear explanations
- Offer relevant practice problems

3. For Practice Sessions:
- Present one question at a time
- Analyze responses
- Adjust difficulty based on performance

Always:
- Use tools proactively to enhance learning
- Provide step-by-step explanations
- Maintain an encouraging tone
- Focus on understanding over memorization

Rules:
- When you need to find a related problem, use tool 1. When present to the user given the search results. Pick one best problem then properly format it to the user.
- When the user provides an answer, use Tool 2 to identify related misconceptions. Analyze the identified misconceptions carefully, considering the relevant thinking tags, and present the most likely misconception to the user. Then, ask if they would like a deeper explanation of the misconception or prefer to move on to a new problem related to it.
"""


@dataclass
class DataLoader:
    """Handle data loading and preprocessing"""

    def __init__(self, train_path: str, misconception_path: str):
        self.df_train_val = pd.read_csv(train_path)
        self.df_miscon = pd.read_csv(misconception_path)
        self.df_train_val_flat = self._preprocess_data()

    def _preprocess_data(self) -> pd.DataFrame:
        return (
            preproc_df(self.df_train_val, self.df_miscon, is_train=True)
            .dropna(subset=["MisconceptionName"])
            .sample(30, random_state=42)
        )


class QuestionFormatter:
    """Format questions with choices and misconceptions"""

    @staticmethod
    def get_misconception_name(df_miscon: pd.DataFrame, miscon_id) -> str:
        if pd.isna(miscon_id):
            return "None"
        return df_miscon.loc[df_miscon.index == miscon_id, "MisconceptionName"].iloc[0]

    @staticmethod
    def format_question(row: pd.Series, df_miscon: pd.DataFrame) -> str:
        try:
            misconceptions = {
                option: QuestionFormatter.get_misconception_name(
                    df_miscon, row[f"Misconception{option}Id"]
                )
                for option in ["A", "B", "C", "D"]
            }

            return f"""Question:
{row["QuestionText"]}
Choices:
A. {row["AnswerAText"]}
B. {row["AnswerBText"]}
C. {row["AnswerCText"]}
D. {row["AnswerDText"]}

Right Answer: {row["CorrectAnswerText"]}

Misconceptions:
""" + "\n".join(
                f"{k}: {v}" for k, v in misconceptions.items()
            )
        except Exception as e:
            print(f"Error processing question {row['QuestionId']}: {e}")
            return None


class State(TypedDict):
    messages: Annotated[list, add_messages]


class TutorAssistant:
    """Main tutor assistant class"""

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        result = self.runnable.invoke(state)
        if not result.tool_calls and (
            not result.content
            or isinstance(result.content, list)
            and not result.content[0].get("text")
        ):
            messages = state["messages"] + [("user", "Respond with a real output.")]
            state = {**state, "messages": messages}
        return {"messages": result}


@tool
def find_related_problems(misconception: str) -> List[str]:
    """Find related problems from the database for a given misconception."""
    return problem_search_engine.search(misconception)


@tool
def find_related_misconceptions(
    question: str,
    correct_answer: str,
    student_answer: str,
    construct_name: str,
    subject_name: str,
) -> List[str]:
    """Find related misconceptions for a given problem."""
    return miscon_search_engine.search_misconception(
        question, correct_answer, student_answer, construct_name, subject_name
    )


def create_workflow(model: ChatOpenAI, tools: List) -> StateGraph:
    """Create and configure the workflow graph"""
    workflow = StateGraph(State)

    primary_assistant = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("placeholder", "{messages}")]
    )

    primary_assistant_runnable = primary_assistant | model.bind_tools(tools)

    workflow.add_node("assistant", TutorAssistant(primary_assistant_runnable))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", "assistant")

    return workflow


def get_graph():
    """Initialize and return the application"""
    # Initialize data
    data_loader = DataLoader("./data/train.csv", "./data/misconception_mapping.csv")

    # Format problems
    problems_with_miscons = (
        data_loader.df_train_val.apply(
            lambda row: QuestionFormatter.format_question(row, data_loader.df_miscon),
            axis=1,
        )
        .dropna()
        .tolist()
    )

    # Initialize search engines
    global miscon_search_engine, problem_search_engine
    miscon_search_engine = SearchMisconceptionEngine(
        documents=data_loader.df_miscon["MisconceptionName"].tolist()
    )
    problem_search_engine = SearchMisconceptionEngine(documents=problems_with_miscons)

    # Create and compile workflow
    model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.0)
    tools = [find_related_problems, find_related_misconceptions]
    workflow = create_workflow(model, tools)
    graph = workflow.compile(checkpointer=MemorySaver())
    from IPython.display import Image, display

    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return graph


# from typing import Annotated, Literal, TypedDict

# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import Runnable, RunnableConfig

# # primary.py
# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from speedy_utils.all import *

# from eedi.client import SearchMisconceptionEngine
# from eedi.data.common import *

# df_train_val = pd.read_csv("./data/train.csv")


# df_miscon = pd.read_csv("./data/misconception_mapping.csv")
# df_train_val_flat = (
#     preproc_df(df_train_val, df_miscon, is_train=True)
#     .dropna(subset=["MisconceptionName"])
#     .sample(30, random_state=42)
# )


# # Using df_miscon instead of df_mapping since df_mapping is not defined in the context
# from textwrap import dedent


# def format_one_question(row):
#     """Format a question with its choices, correct answer, and misconceptions"""
#     try:
#         # Get misconception names safely
#         misconceptions = {
#             "A": (
#                 df_miscon.loc[
#                     df_miscon.index == row["MisconceptionAId"], "MisconceptionName"
#                 ].iloc[0]
#                 if pd.notna(row["MisconceptionAId"])
#                 else "None"
#             ),
#             "B": (
#                 df_miscon.loc[
#                     df_miscon.index == row["MisconceptionBId"], "MisconceptionName"
#                 ].iloc[0]
#                 if pd.notna(row["MisconceptionBId"])
#                 else "None"
#             ),
#             "C": (
#                 df_miscon.loc[
#                     df_miscon.index == row["MisconceptionCId"], "MisconceptionName"
#                 ].iloc[0]
#                 if pd.notna(row["MisconceptionCId"])
#                 else "None"
#             ),
#             "D": (
#                 df_miscon.loc[
#                     df_miscon.index == row["MisconceptionDId"], "MisconceptionName"
#                 ].iloc[0]
#                 if pd.notna(row["MisconceptionDId"])
#                 else "None"
#             ),
#         }

#         return f"""Question:
# {row["QuestionText"]}
# Choices:
# A. {row["AnswerAText"]}
# B. {row["AnswerBText"]}
# C. {row["AnswerCText"]}
# D. {row["AnswerDText"]}

# Right Answer: {row["CorrectAnswerText"]}

# Misconceptions:
# A: {misconceptions['A']}
# B: {misconceptions['B']}
# C: {misconceptions['C']}
# D: {misconceptions['D']}"""
#     except Exception as e:
#         print(f"Error processing question {row['QuestionId']}: {e}")
#         return None


# # Apply the formatting function to create problems with misconceptions
# problems_with_miscons = (
#     df_train_val.apply(format_one_question, axis=1).dropna().tolist()
# )


# miscon_search_engine = SearchMisconceptionEngine(
#     documents=df_miscon["MisconceptionName"].tolist()
# )
# problem_search_engine = SearchMisconceptionEngine(documents=problems_with_miscons)


# class State(TypedDict):
#     messages: Annotated[list, add_messages]


# # Define the tools for the agent to use
# class Assistant:
#     def __init__(self, runnable: Runnable):
#         self.runnable = runnable

#     def __call__(self, state: State, config: RunnableConfig):
#         result = self.runnable.invoke(state)

#         # If the result is empty, prompt the user to provide a real output
#         if not result.tool_calls and (
#             not result.content
#             or isinstance(result.content, list)
#             and not result.content[0].get("text")
#         ):
#             messages = state["messages"] + [("user", "Respond with a real output.")]
#             state = {**state, "messages": messages}
#         return {"messages": result}


# @tool
# def find_related_problems(misconception: str) -> List[str]:
#     """
#     Find related problems from the database that address a given misconception.
#     This function searches through a problem database to find relevant practice problems
#     """
#     return problem_search_engine.search(misconception)


# @tool
# def find_related_misconceptions(
#     question: str,
#     correct_answer: str,
#     student_answer: str,
#     construct_name: str,
#     subject_name: str,
# ) -> List[str]:
#     """
#     Find related misconceptions from the database that are addressed by a given problem.
#     This function searches through a misconception database to find relevant misconceptions
#     """
#     mocks = miscon_search_engine.search_misconception(
#         question, correct_answer, student_answer, construct_name, subject_name
#     )
#     return mocks


# def get_app():
#     tools = [find_related_problems, find_related_misconceptions]

#     tool_node = ToolNode(tools)

#     model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.0)

#     primary_assistant = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are an AI math tutor with access to specialized tools for personalized instruction. Your primary goal is to help students master mathematical concepts through targeted practice and misconception correction.

#     Available Tools:
#     1. find_related_problems(misconception: str)
#     - Use this to fetch practice problems when:
#     - Student shows confusion about a concept
#     - Additional practice is needed
#     - Reviewing specific topics

#     2. find_related_misconceptions(text_problem, answer_text, correct_answer)
#     - Use this to analyze student responses and identify misconceptions
#     - Help diagnose learning gaps
#     - Guide remedial instruction

#     Teaching Protocol:
#     1. For New Topics:
#     - Assess current understanding
#     - Explain core concepts
#     - Use find_related_problems for targeted practice

#     2. When Student Makes Mistakes:
#     - Use find_related_misconceptions to diagnose issues
#     - Provide clear explanations
#     - Offer relevant practice problems

#     3. For Practice Sessions:
#     - Present one question at a time
#     - Analyze responses
#     - Adjust difficulty based on performance

#     Always:
#     - Use tools proactively to enhance learning
#     - Provide step-by-step explanations
#     - Maintain an encouraging tone
#     - Focus on understanding over memorization

#     Rules:
#     - When you need to find a related problem, use tool 1. When present to the user given the search results. Pick one best problem then properly format it to the user.
#     - When the user provides an answer, use Tool 2 to identify related misconceptions. Analyze the identified misconceptions carefully, considering the relevant thinking tags, and present the most likely misconception to the user. Then, ask if they would like a deeper explanation of the misconception or prefer to move on to a new problem related to it.
#     """,
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     )
#     primary_assistant_runable = primary_assistant | model.bind_tools(tools)

#     # Define a new graph
#     workflow = StateGraph(State)

#     # Define the two nodes we will cycle between
#     workflow.add_node("assistant", Assistant(primary_assistant_runable))
#     workflow.add_node("tools", tool_node)

#     workflow.add_edge(START, "assistant")

#     # We now add a conditional edge
#     workflow.add_conditional_edges(
#         "assistant",
#         tools_condition,
#     )

#     workflow.add_edge("tools", "assistant")

#     checkpointer = MemorySaver()

#     app = workflow.compile(checkpointer=checkpointer)
#     from IPython.display import Image

#     display(Image(app.get_graph(xray=True).draw_mermaid_png()))
#     # Use the Runnable

#     return app
