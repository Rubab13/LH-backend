from pydantic import BaseModel
from datetime import datetime
from typing import List

class LoginModel(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class QueryRequest(BaseModel):
    query: str
    id_user: int

class QAResponse(BaseModel):
    question: str
    answer: str

class SignupModel(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str
    confirmPassword: str
    state: str
    isLegalProfessional: bool

class HistoryItem(BaseModel):
    question: str
    answer: str
    timestamp: datetime
