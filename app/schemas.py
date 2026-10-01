from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, description="Pergunta em linguagem natural sobre o catálogo.")


class AskResponse(BaseModel):
    pergunta: str
    resposta: str
    sql_executado: list[str] = Field(
        default_factory=list, description="SQL(s) gerado(s) e executado(s) pelo agente, para transparência/debug."
    )
