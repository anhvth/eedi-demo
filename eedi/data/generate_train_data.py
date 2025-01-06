"""
This module provides functions and classes to generate training data for math problems based on specific misconceptions. 
It includes utilities to format input data, construct prompts, and generate examples for training models to identify and address student misconceptions.

Functions:
    construct_base_prompt_v3(row, wrong_choice):
        Constructs a base prompt for a given row of data and a specified wrong answer choice.
    
    format_input_v3(row, wrong_choice):
        Formats the input data for a given row and wrong answer choice, including additional metadata.

    get_dspy_example(row, wrong_choice):
        Generates an example for the dspy library based on the given row and wrong answer choice.

Classes:
    SignatureFillReasoningAndPlan(dspy.Signature):
        A dspy signature class to generate a plausible math problem based on a given misconception, including fields for reasoning and planning.

    SignatureProblemGenerator(dspy.Signature):
        A dspy signature class to generate a plausible math problem based on a given misconception, including fields for planning and generating the question, correct answer, and student wrong answer.

Constants:
    TEMPLATE_INPUT_V3:
        A template string for formatting the input data for version 3.
    
    ConstructNameDesc:
        A description of construct names used in the module.
    
    SubjectNameDesc:
        A description of subject names used in the module.

"""

import os
import random
# from dotenv import load_dotenv
import pandas as pd
from speedy_utils.all import *
import dspy
from llm_utils import *

# from eedi.common import *
# from eedi.format_data import format_input_v2, format_input_v3

# load_dotenv()
setup_logger("E")

TEMPLATE_INPUT_V3 = """{QUESTION}\nCorrect text: {CORRECT_ANSWER}\nStudent wrong answer: {STUDENT_WRONG_ANSWER}"""


def construct_base_prompt_v3(row, wrong_choice):
    assert wrong_choice in "ABCD", "Invalid choice for wrong answer."

    # Extract values from the row
    question_text = row.get("QuestionText", "No question text provided")
    subject_name = row.get("SubjectName", "Unknown subject")
    construct_name = row.get("ConstructName", "Unknown construct")

    correct_answer = row.get("CorrectAnswer", "Unknown")
    assert wrong_choice != correct_answer, "Wrong choice cannot be the correct answer."

    correct_answer_text = row.get(
        f"Answer{correct_answer}Text", "No correct answer text available"
    )
    wrong_answer_text = row.get(
        f"Answer{wrong_choice}Text", "No wrong answer text available"
    )

    formatted_question = (
        f"Question: {question_text}\n\n"
        f"SubjectName: {subject_name}\n"
        f"ConstructName: {construct_name}"
    )
    return TEMPLATE_INPUT_V3.format(
        QUESTION=formatted_question,
        CORRECT_ANSWER=correct_answer_text,
        STUDENT_WRONG_ANSWER=wrong_answer_text,
    )


def format_input_v3(row, wrong_choice):
    global df_mapping, df_train

    if isinstance(row, int):
        row = df_train.iloc[row]
    correct_answer = row.get("CorrectAnswer", "Unknown")
    assert wrong_choice != correct_answer
    correct_answer_text = row.get(
        f"Answer{correct_answer}Text", "No correct answer text available"
    )
    wrong_answer_text = row.get(
        f"Answer{wrong_choice}Text", "No wrong answer text available"
    )
    mid = row[f"Misconception{wrong_choice}Id"]
    if not pd.isna(mid):
        mid = int(mid)
    ret = {
        "PROMPT": construct_base_prompt_v3(row, wrong_choice),
        "QUESTION": row["QuestionText"],
        "CORRECT_ANSWER": correct_answer_text,
        "STUDENT_WRONG_ANSWER": wrong_answer_text,
        "MISCONCEPTION_NAME": (
            df_mapping.loc[mid]["MisconceptionName"] if isinstance(mid, int) else None
        ),
        "MISCONCEPTION_ID": mid,
    }
    return ret


ConstructNameDesc = """
Construct names are detailed descriptions of specific mathematical skills or concepts that students are expected to understand and master. They outline precise tasks or abilities, such as performing arithmetic operations, simplifying algebraic expressions, calculating statistical measures, solving equations, interpreting graphs, and applying geometric properties. Each construct focuses on a specific learning objective within the mathematics curriculum and represents a particular concept or procedure that can be taught or assessed.

Examples of construct names:
- Use the order of operations to carry out calculations involving powers.
- Simplify an algebraic fraction by factorising the numerator.
- Calculate the range from a list of data.
- Recall and use the intersecting diagonals properties of a rectangle.
- Substitute positive integer values into formulae involving powers or roots.
"""

SubjectNameDesc = """
Subject names represent broader mathematical topics or domains under which various constructs are categorized. These subjects encompass areas such as Arithmetic, Algebra, Geometry, Measurement and Conversion, Data Handling, Graphical Representation, and Problem Solving. Each subject groups together related concepts and skills, facilitating structured learning, teaching, and assessment.

Examples of subject names:
- BIDMAS (Order of Operations)
- Simplifying Algebraic Fractions
- Range and Interquartile Range from a List of Data
- Properties of Quadrilaterals
- Substitution into Formula
- Area of Simple Shapes
- Converting between Fractions and Percentages
- Multiplying and Dividing with Decimals
"""


def get_dspy_example(row, wrong_choice):
    # Example consistst of PROMPT | misconception | missing reasoning and plan
    item = format_input_v3(row, wrong_choice=wrong_choice)
    return dspy.Example(
        base=SignatureProblemGenerator,
        **item,
        reasoning="MISSING_VALUE",
        PLANING="MISSING_VALUE",
    )


class SignatureFillReasoningAndPlan(dspy.Signature):
    """Generate a plausible math problem based on a given misconception.

    Here is some meta data for the construct name and subject name for you
    <META_DATA>
    <CONSTRUCT_NAME>
    Examples of construct names:
        - Use the order of operations to carry out calculations involving powers.
        - Simplify an algebraic fraction by factorising the numerator.
        - Calculate the range from a list of data.
        - Recall and use the intersecting diagonals properties of a rectangle.
        - Substitute positive integer values into formulae involving powers or roots.
        - ...
    </CONSTRUCT_NAME>
    <SUBJECT_NAME>
        - BIDMAS (Order of Operations)
        - Simplifying Algebraic Fractions
        - Range and Interquartile Range from a List of Data
        - Properties of Quadrilaterals
        - Substitution into Formula
        - Area of Simple Shapes
        - Converting between Fractions and Percentages
        - Multiplying and Dividing with Decimals
        - ...
    </SUBJECT_NAME>
    </META_DATA>

    NOTE: Be *creative* when generate the questions. Just make sure it align with the *given misconception*
    """

    MISCONCEPTION_NAME = dspy.InputField(
        desc="A specific misconception related to a mathematical concept"
    )
    QUESTION = dspy.InputField(
        desc="A generated math problem consist of Question, SubjectName, and ConstructName"
    )
    CORRECT_ANSWER = dspy.InputField(
        desc="The correct answer to the generated math problem."
    )
    STUDENT_WRONG_ANSWER = dspy.InputField(
        desc="An incorrect answer that a student might give based on the misconception."
    )

    reasoning = dspy.OutputField()
    PLANING = dspy.OutputField(
        desc="This is where you create a plan for generating question/correct/incorrect answer before actually generating it. Focus on the input field to generate a relevant plan, evaluate your plan and make changes all in here"
    )


class SignatureProblemGenerator(dspy.Signature):
    """
    Generate a plausible math problem based on a given misconception.

    Here is some meta data for the construct name and subject name for you
    <META_DATA>
    <CONSTRUCT_NAME>
    Examples of construct names:
        - Use the order of operations to carry out calculations involving powers.
        - Simplify an algebraic fraction by factorising the numerator.
        - Calculate the range from a list of data.
        - Recall and use the intersecting diagonals properties of a rectangle.
        - Substitute positive integer values into formulae involving powers or roots.
        - ...
    </CONSTRUCT_NAME>
    <SUBJECT_NAME>
        - BIDMAS (Order of Operations)
        - Simplifying Algebraic Fractions
        - Range and Interquartile Range from a List of Data
        - Properties of Quadrilaterals
        - Substitution into Formula
        - Area of Simple Shapes
        - Converting between Fractions and Percentages
        - Multiplying and Dividing with Decimals
    </SUBJECT_NAME>
    </META_DATA>

    NOTE: Be *creative* when generate the questions. Just make sure it align with the *given misconception*
    """

    MISCONCEPTION_NAME = dspy.InputField(
        desc="A specific misconception related to a mathematical concept"
    )
    PLANING = dspy.OutputField(
        desc="This is where you create a plan for generating question/correct/incorrect answer before actually generating it. Focus on the input field to generate a relevant plan, evaluate your plan and make changes all in here"
    )
    QUESTION = dspy.OutputField(
        desc="A generated math problem consist of Question, SubjectName, and ConstructName"
    )
    CORRECT_ANSWER = dspy.OutputField(
        desc="The correct answer to the generated math problem."
    )
    STUDENT_WRONG_ANSWER = dspy.OutputField(
        desc="An incorrect answer that a student might give based on the misconception."
    )
