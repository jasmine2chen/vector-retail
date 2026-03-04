from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent.bank_grade_agent import build_bank_graph, AgentState
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)
agent_graph = build_bank_graph()

class QueryRequest(BaseModel):
    query: str
    thread_id: str

@app.post("/api/v1/query")
async def query_agent(req: QueryRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = AgentState(request_id=req.thread_id, query=req.query, messages=[])
    try:
        final = await agent_graph.ainvoke(initial_state, config=config)
        if final["verification_status"] == "rejected":
            return {"status": "blocked", "reason": final["rejection_reason"]}
        return {
            "status": "success",
            "answer": final["messages"][-1].content,
            "citations": [d.doc_id for d in final["valid_docs"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
