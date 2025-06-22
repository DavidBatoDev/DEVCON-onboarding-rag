from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
