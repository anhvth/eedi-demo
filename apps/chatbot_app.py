from apps.personalize_math_tutor_graph import get_graph
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

graph = get_graph()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": 42}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    final_state = graph.invoke(
        {
            "messages": [
                ("human", msg.content),
            ]
        },
        config={"configurable": {"thread_id": 42}},
    )
    final_answer.content = final_state["messages"][-1].content

    await final_answer.send()
