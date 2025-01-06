import datetime
import os
import re
from datetime import datetime
from typing import List

import dspy
import pandas as pd
from loguru import logger


from typing import List, Optional

import numpy as np
from tqdm import tqdm

TEMPLATE_INPUT_V3 = """{QUESTION}\nCorrect text: {CORRECT_ANSWER}\nStudent wrong answer: {STUDENT_WRONG_ANSWER}"""


def extract_tag(text: str, tag: str) -> str:
    """Extract content within a specified XML-like tag."""
    pattern = rf"<{tag}.*?>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def preproc_df(df_train_or_val, df_mapping, is_train):
    def get_correct_answer(row):
        if row["CorrectAnswer"] == "A":
            return row["AnswerAText"]
        elif row["CorrectAnswer"] == "B":
            return row["AnswerBText"]
        elif row["CorrectAnswer"] == "C":
            return row["AnswerCText"]
        elif row["CorrectAnswer"] == "D":
            return row["AnswerDText"]
        else:
            return None

    # Apply the function to create the CorrectAnswer column
    df_train_or_val["CorrectAnswerText"] = df_train_or_val.apply(
        get_correct_answer, axis=1
    )
    INPUT_TEMPLATE = """- **Question**: {Question}
- **Incorrect Answer**: {IncorrectAnswer}
- **Correct Answer**: {CorrectAnswer}
- **Construct Name**: {ConstructName}
- **Subject Name**: {SubjectName}
"""

    def apply_template(row):
        return INPUT_TEMPLATE.format(
            ConstructName=row["ConstructName"],
            SubjectName=row["SubjectName"],
            Question=row["QuestionText"],
            IncorrectAnswer=row[f"AnswerText"],
            CorrectAnswer=row[f"CorrectAnswerText"],
        )

    # Define the columns to select
    selected_columns = [
        "QuestionId",
        "ConstructName",
        "SubjectName",
        "CorrectAnswer",
        "QuestionText",
        "CorrectAnswerText",
    ]

    # Prepare a list to collect the rows
    rows = []
    # Loop through each row in the dataframe
    for _, row in df_train_or_val.iterrows():
        # Extract static data for the row
        static_data = [row[col] for col in selected_columns]

        # Iterate through each answer option (A, B, C, D)
        for option in ["A", "B", "C", "D"]:
            # Construct option name and corresponding answer text
            option_text_column = f"Answer{option}Text"
            answer_text = row[option_text_column]
            if static_data[-1] == answer_text:
                continue
            if is_train:
                # Get the misconception corresponding to the current option
                misconception = row[f"Misconception{option}Id"]
            else:
                misconception = None
            # Append the data to the rows list
            row_data = static_data + [option, answer_text, misconception]
            rows.append(row_data)
    df_answer = pd.DataFrame(
        rows, columns=selected_columns + ["Option", "AnswerText", "MisconceptionId"]
    )

    # Create a new dataframe from the collected rows

    # df_answer = df_answer[~df_answer["MisconceptionId"].isna()]
    if is_train:
        df_answer["MisconceptionName"] = df_answer["MisconceptionId"].apply(
            lambda x: (
                df_mapping.loc[int(x)]["MisconceptionName"] if not pd.isna(x) else None
            )
        )

    df_answer["Prompt"] = df_answer.apply(lambda row: apply_template(row), axis=1)
    # ----
    return df_answer


mapping_path = "data/misconception_mapping.csv"
df_mapping = pd.read_csv(mapping_path)


def get_df_parsed(
    df_path="data/train.csv",
):
    df_train = pd.read_csv(df_path)

    df_answer = preproc_df(df_train, df_mapping, True)
    return df_answer


# Updated function to work for any DataFrame
def format_df_generic(df):
    formatted_output = ""
    for idx, row in df.iterrows():
        formatted_output += f"Row ID: {idx}. "
        formatted_output += ", ".join([f"{col}: {row[col]}" for col in df.columns])
        formatted_output += "\n"
    return formatted_output


class VectorDBRetriever(dspy.Retrieve):
    def __init__(self, db):
        self.db = db

    def search(self, query, k):
        results = self.db.search(query, k)
        return [
            f'MisconceptionId={doc.metadata["MisconceptionId"]} | {doc.page_content}'
            for doc, _ in results
        ]


trainset = None


def prepare_db_train_val():
    df = get_df_parsed()

    # vector_db = VectorDBRetriever(db)
    # Prepare the Training Data
    trainset = []
    devset = []
    for index, row in df.iterrows():
        if pd.isna(row["MisconceptionId"]):
            continue
        example = dspy.Example(
            input_string=row["Prompt"],
            target_misconception=row["MisconceptionName"],
            target_misconception_id=row["MisconceptionId"],
        )
        if row["MisconceptionId"] % 10 == 0:
            devset.append(example)
        else:
            trainset.append(example)

    trainset = [x.with_inputs("input_string") for x in trainset]
    devset = [x.with_inputs("input_string") for x in devset]
    vector_db, db = None, None
    return trainset, devset, vector_db, db


if not trainset:
    # trainset, devset, vector_db, db = None, None, None, None
    trainset, devset, vector_db, db = prepare_db_train_val()


def compute_metric(
    query_embeddings: List[List[float]],
    document_embeddings: List[List[float]],
    expected_ids_per_query: List[List[int]],
    k: int = 25,
) -> float:
    """
    Computes the Mean Average Precision (MAP) at K.

    :param query_embeddings: A list of query embeddings.
    :param document_embeddings: A list of document embeddings.
    :param expected_ids_per_query: A list of lists where each inner list contains the IDs of relevant documents for a query.
    :param k: The number of top documents to consider when calculating precision.
    :return: The Mean Average Precision (MAP) score.
    """

    # Convert the embeddings to numpy arrays for easier manipulation
    query_embeddings = np.array(query_embeddings)
    document_embeddings = np.array(document_embeddings)

    num_queries = len(query_embeddings)
    average_precisions = []
    average_recall_at_k = []
    # Iterate over each query
    for i in tqdm(range(num_queries)):
        query_embed = query_embeddings[i]
        expected_ids = set(
            expected_ids_per_query[i]
        )  # Set of relevant document IDs for the query

        # Compute similarity between the query and all document embeddings (dot product)
        similarities = np.dot(query_embed, document_embeddings.T)

        # Get the indices of the documents sorted by similarity (descending order)
        ranked_doc_indices = np.argsort(-similarities)

        # Calculate Average Precision (AP) for this query
        num_relevant = 0
        precision_at_k = []

        for rank, doc_idx in enumerate(ranked_doc_indices[:k]):
            if doc_idx in expected_ids:
                num_relevant += 1
                precision = num_relevant / (rank + 1)  # Precision at this rank
                precision_at_k.append(precision)

        if precision_at_k:
            average_precision = np.mean(precision_at_k)
        else:
            average_precision = 0.0
        average_recall_at_k.append(average_precision > 0)
        average_precisions.append(average_precision)

    # Compute the Mean Average Precision (MAP)
    mean_average_precision = np.mean(average_precisions)
    mean_averge_recall_at_k = np.mean(average_recall_at_k)
    return mean_average_precision, mean_averge_recall_at_k


def mine_hard_negatives(
    query_embeddings: List[List[float]],
    document_embeddings: List[List[float]],
    expected_ids_per_query: List[List[int]],
    k: int = 25,
    query_texts: Optional[List[str]] = None,
    document_texts: Optional[List[str]] = None,
) -> List[dict]:
    """
    Mines hard negatives for each query, with optional query and document texts.

    :param query_embeddings: A list of query embeddings.
    :param document_embeddings: A list of document embeddings.
    :param expected_ids_per_query: A list of lists where each inner list contains the IDs of relevant documents for a query.
    :param k: The number of hard negatives to mine for each query.
    :param query_texts: (Optional) A list of query texts corresponding to the query embeddings.
    :param document_texts: (Optional) A list of document texts corresponding to the document embeddings.
    :return: A list of dictionaries, each containing the query, its positive (relevant) documents, and hard negatives.
             If query_texts and document_texts are provided, the text for each query, positive document, and hard negative will also be returned.
    """

    # Convert the embeddings to numpy arrays for easier manipulation
    query_embeddings = np.array(query_embeddings)
    document_embeddings = np.array(document_embeddings)

    num_queries = len(query_embeddings)
    hard_negatives_data = []

    # Iterate over each query
    for i in tqdm(range(num_queries)):
        query_embed = query_embeddings[i]
        expected_ids = set(
            expected_ids_per_query[i]
        )  # Set of relevant document IDs for the query

        # Compute similarity between the query and all document embeddings (dot product)
        similarities = np.dot(query_embed, document_embeddings.T)

        # Get the indices of the documents sorted by similarity (descending order)
        ranked_doc_indices = np.argsort(-similarities)

        # Filter out the positive examples (relevant documents)
        hard_negatives = []
        for doc_idx in ranked_doc_indices:
            if doc_idx not in expected_ids:
                hard_negatives.append(doc_idx)
            if len(hard_negatives) == k:
                break

        # Create the result dictionary
        result = {
            "query": i,  # Index of the query
            "pos": list(expected_ids),  # List of positive (relevant) document IDs
            "neg": hard_negatives,  # List of hard negatives (IDs)
        }
        if query_texts and document_texts:
            # turn id to tuple id,text
            result["query"] = [i, query_texts[i]]
            result["pos"] = [(j, document_texts[j]) for j in expected_ids][0]
            result["neg"] = [(j, document_texts[j]) for j in hard_negatives]

        hard_negatives_data.append(result)

    return hard_negatives_data


def update_train_df(df):
    # format train_mmdd_hh_mm.csv
    assert len(df) == 5607
    os.makedirs("./data/train_parquets", exist_ok=True)
    df_len = len(df)
    output_path = f"./data/train_parquets/{datetime.now().strftime('%m%d_%H_%M')}_len{df_len}.parquet"
    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


def load_latest_df():
    paths = os.listdir("./data/train_parquets")
    paths.sort()
    latest_path = paths[-1]
    df = pd.read_parquet(f"./data/train_parquets/{latest_path}")
    column_comma = df.columns.to_list()
    column_comma = "\n".join(column_comma)
    print(f"Loading from {latest_path}, columns: ```\n{column_comma}```")
    # ignore col QUESTION	CORRECT_ANSWER	STUDENT_WRONG_ANSWER
    for col in [
        "QUESTION",
        "CORRECT_ANSWER",
        "STUDENT_WRONG_ANSWER",
        "ConstructName",
        "SubjectName",
        "MISCONCEPTION_ID",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


__all__ = [
    "VectorDBRetriever",
    "prepare_db_train_val",
    "preproc_df",
    "get_df_parsed",
    "format_df_generic",
    "compute_metric",
    "mine_hard_negatives",
    "mine_hard_negatives",
    "extract_tag",
    "df_mapping",
    "TEMPLATE_INPUT_V3",
    # "construct_base_prompt_v3",
    # "get_synthetic_df",
    # "format_docs",
    # "get_parsed_df_v3",
    # "get_parsed_df_v4",
    # "get_openai_misconception_db",
    # "update_train_df",
    # "load_latest_df",
    # "calculate_similarity_misconception",
    # "OpenAIModel",
    # "LocalLM",
    # "get_df_train_val",
    # "format_input_v3",
]

logger.info(f"common: {__all__}")
