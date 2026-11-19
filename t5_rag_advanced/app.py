import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from commons.constants import (
    OPENAI_API_KEY,
    OPENAI_CHAT_EMBEDDING_ENDPOINT,
    OPENAI_EMBEDDING_ENDPOINT,
    OPENAI_EMBEDDINGS_MODEL,
    OPENAI_TERRA_MODEL,
)
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import SearchMode, TextProcessor

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


def main() -> None:
    embeddings_client = EmbeddingsClient(
        endpoint=f"{OPENAI_EMBEDDING_ENDPOINT}/embeddings",
        model_name=OPENAI_EMBEDDINGS_MODEL,
        api_key=OPENAI_API_KEY,
    )
    completion_client = ChatCompletionClient(
        endpoint=f"{OPENAI_CHAT_EMBEDDING_ENDPOINT}/chat/completions",
        model_name=OPENAI_TERRA_MODEL,
        api_key=OPENAI_API_KEY,
    )
    text_processor = TextProcessor(
        embeddings_client=embeddings_client,
        db_config={
            "host": "localhost",
            "port": 5433,
            "database": "vectordb",
            "user": "postgres",
            "password": "postgres",
        },
    )

    print("🎯 Microwave RAG Assistant")
    print("=" * 100)
    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
    if load_context.lower() in {"y", "yes"}:
        text_processor.process_text_file(
            file_name=str(PROJECT_ROOT / "t5_rag_advanced/embeddings/microwave_manual.txt"),
            chunk_size=400,
            overlap=40,
            dimensions=384,
        )
        print("=" * 100)

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_request = input("\n➡️ ").strip()

        if user_request.lower() in {"quit", "exit"}:
            print("👋 Goodbye")
            break
        if not user_request:
            continue

        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        context = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_request,
            top_k=5,
            score_threshold=0.5,
            dimensions=384,
        )

        print(f"\n{'=' * 100}\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
        augmented_prompt = USER_PROMPT.format(
            context="\n\n".join(context), query=user_request
        )
        conversation.add_message(Message(Role.USER, augmented_prompt))
        print(f"Prompt:\n{augmented_prompt}")

        print(f"\n{'=' * 100}\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        ai_message = completion_client.get_completion(conversation.get_messages())
        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)
        conversation.add_message(ai_message)


if __name__ == "__main__":
    main()
