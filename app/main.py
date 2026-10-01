from fastapi import FastAPI, HTTPException

from app.agent import ask
from app.schemas import AskRequest, AskResponse

app = FastAPI(
    title="CineData Analytics",
    description="Agente Text-to-SQL para perguntas em linguagem natural sobre o catálogo de filmes.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    try:
        resposta, sqls = ask(payload.pergunta)
    except Exception as exc:  # noqa: BLE001 - devolve qualquer falha do agente como 500 legível
        raise HTTPException(status_code=500, detail=f"Falha ao processar a pergunta: {exc}") from exc

    return AskResponse(pergunta=payload.pergunta, resposta=resposta, sql_executado=sqls)
