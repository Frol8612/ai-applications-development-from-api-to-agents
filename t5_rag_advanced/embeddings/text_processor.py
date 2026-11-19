from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict[str, Any]):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
        self,
        file_name: str,
        chunk_size: int,
        overlap: int,
        dimensions: int,
        truncate_table: bool = True,
    ) -> None:
        """Chunk a UTF-8 text file, embed it, and store the resulting vectors."""
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("Overlap must be non-negative and smaller than chunk size")
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than zero")

        file_path = Path(file_name)
        content = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(content, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"File contains no text: {file_path}")

        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)
        missing_indexes = set(range(len(chunks))) - embeddings.keys()
        if missing_indexes:
            raise ValueError(
                f"Embeddings response is missing indexes: {sorted(missing_indexes)}"
            )

        connection = self._get_connection()
        try:
            if truncate_table:
                self._truncate_table(connection)
            for index, chunk in enumerate(chunks):
                self._save_chunk(
                    document_name=file_path.name,
                    text=chunk,
                    embedding=embeddings[index],
                    connection=connection,
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _truncate_table(self, connection=None) -> None:
        """Remove all stored vectors, reusing an existing connection when provided."""
        owns_connection = connection is None
        connection = connection or self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
            if owns_connection:
                connection.commit()
        except Exception:
            if owns_connection:
                connection.rollback()
            raise
        finally:
            if owns_connection:
                connection.close()

    def _save_chunk(
        self,
        document_name: str,
        text: str,
        embedding: list[float],
        connection=None,
    ) -> None:
        """Store a chunk and its embedding, optionally in an existing transaction."""
        owns_connection = connection is None
        connection = connection or self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO vectors (document_name, text, embedding)
                    VALUES (%s, %s, %s::vector)
                    """,
                    (document_name, text, str(embedding)),
                )
            if owns_connection:
                connection.commit()
        except Exception:
            if owns_connection:
                connection.rollback()
            raise
        finally:
            if owns_connection:
                connection.close()

    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int,
        score_threshold: float,
        dimensions: int,
    ) -> list[str]:
        """Return the closest stored chunks whose distance meets the threshold."""
        if not isinstance(search_mode, SearchMode):
            raise ValueError("Unsupported search mode")
        if not user_request or not user_request.strip():
            raise ValueError("User request cannot be empty")
        if top_k <= 0:
            raise ValueError("Top k must be greater than zero")
        if score_threshold < 0:
            raise ValueError("Score threshold cannot be negative")
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than zero")

        embedding_data = self.embeddings_client.get_embeddings(
            user_request, dimensions
        )
        query_embedding = embedding_data.get(0)
        if query_embedding is None:
            raise ValueError("Embeddings response is missing index 0")

        operator = (
            "<->"
            if search_mode == SearchMode.EUCLIDIAN_DISTANCE
            else "<=>"
        )
        query = f"""
            SELECT text
            FROM (
                SELECT
                    text,
                    embedding {operator} %s::vector AS distance
                FROM vectors
            ) AS ranked
            WHERE distance <= %s
            ORDER BY distance
            LIMIT %s
        """

        connection = self._get_connection()
        try:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    query,
                    (str(query_embedding), score_threshold, top_k),
                )
                return [row["text"] for row in cursor.fetchall()]
        finally:
            connection.close()
