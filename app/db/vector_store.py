from datetime import date
from typing import List
from pydantic import BaseModel

class Document(BaseModel):
    doc_id: str
    content: str
    source: str
    page: int
    retrieval_score: float
    date_filed: date

def fetch_documents(query: str, top_k: int = 3) -> List[Document]:
    \"\"\"Mock Data Access Layer. In production, connects to Pinecone/Weaviate.\"\"\"
    return [
        Document(
            doc_id="10k_2024_fy", 
            content="Total Revenue for FY2024 was $50.0 million. Net Income was $12.5 million.", 
            source="10-K", page=40, retrieval_score=0.95, date_filed=date(2024, 3, 1)
        ),
        Document(
            doc_id="10q_2023_q3", 
            content="Operating expenses were $30 million.", 
            source="10-Q", page=12, retrieval_score=0.75, date_filed=date(2023, 10, 15)
        )
    ]
