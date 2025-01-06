from textwrap import dedent
import requests
import numpy as np
from speedy_utils import memoize


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}"


def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}\n<response>{response}"


def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len
        - len(tokenizer("<s>", add_special_tokens=False)["input_ids"])
        - len(tokenizer("\n<response></s>", add_special_tokens=False)["input_ids"]),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer("\n<response>", add_special_tokens=False)["input_ids"]
    new_max_length = (
        len(prefix_ids) + len(suffix_ids) + query_max_len + 8
    ) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs["input_ids"])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + "\n<response>"
    return new_max_length, new_queries


def build_llm_query(
    question,
    correct_answer,
    student_wrong_answer,
    construct_name,
    subject_name,
    task=None,
):
    """
    Build a query for the language model.

    Args:
        question (str): The question text.
        correct_answer (str): The correct answer text.
        student_wrong_answer (str): The student's wrong answer text.
        construct_name (str): The construct name.
        subject_name (str): The subject name.
        task (str, optional): The task description. Defaults to a predefined task.

    Returns:
        str: The formatted query for the language model.
    """
    INPUT_TEMPLATE = dedent(
        """
                    - **Question**: {Question}
                    - **Incorrect Answer**: {IncorrectAnswer}
                    - **Correct Answer**: {CorrectAnswer}
                    - **Construct Name**: {ConstructName}
                    - **Subject Name**: {SubjectName}
                    """
    )

    raw_query = INPUT_TEMPLATE.format(
        ConstructName=construct_name,
        SubjectName=subject_name,
        Question=question,
        IncorrectAnswer=student_wrong_answer,
        CorrectAnswer=correct_answer,
    )

    if task is None:
        task = "Given a math multiple-choice problem with a student's wrong answer, retrieve the math misconceptions"

    llm_query = get_detailed_instruct(task, raw_query)
    return llm_query


class SearchMisconceptionEngine:
    def __init__(self, url="http://0.0.0.0:8080/v1/embeddings", documents=None):
        self.url = url
        if documents is not None:
            self.set_documents(documents)

    def set_documents(self, documents):
        """
        Set self.documents and self.docuemnts_embeddings
        """
        self.documents = documents
        self.documents_embeddings = self.get_embeddings(documents)

    def get_embeddings(self, texts):
        """
        Get embeddings for a list of texts from the local embedding service

        Args:
             texts (list): List of strings to get embeddings for

        Returns:
             np.ndarray: Array of embeddings
        """
        @memoize
        def _f(texts):
            payload = {"texts": texts}
            response = requests.post(self.url, json=payload)
            response = response.json()
            return np.array(response["embeddings"])
        return _f(texts)

    def search(self, prompt, topk=10):
        response = requests.post(self.url, json={"texts": [prompt]})
        embedings = response.json()["embeddings"]

        # find the most similar document
        similarities = np.dot(self.documents_embeddings, embedings[0])
        topk_indices = similarities.argsort()[-topk:][::-1]
        top_misconceptions = [self.documents[i] for i in topk_indices]
        return top_misconceptions

    def search_misconception(
        self,
        question,
        correct_answer,
        student_wrong_answer,
        construct_name,
        subject_name,
        task=None,
    ):
        """
        Search for misconceptions using the language model.

        Args:
            question (str): The question text.
            correct_answer (str): The correct answer text.
            student_wrong_answer (str): The student's wrong answer text.
            construct_name (str): The construct name.
            subject_name (str): The subject name.
            task (str, optional): The task description. Defaults to a predefined task.

        Returns:
            dict: The response from the language model
        """
        query = build_llm_query(
            question,
            correct_answer,
            student_wrong_answer,
            construct_name,
            subject_name,
            task,
        )

        return self.search(query)
