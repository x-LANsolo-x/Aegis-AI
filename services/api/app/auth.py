from __future__ import annotations

from typing import List

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

# Role constants
ROLE_FIELD = "FIELD"
ROLE_ANALYST = "ANALYST"
ROLE_ADMIN = "ADMIN"


class User(BaseModel):
    id: str
    roles: List[str] = Field(default_factory=list)


def get_current_user(request: Request) -> User:
    return request.state.user


def require_role(role: str):
    def checker(user: User = Depends(get_current_user)):
        if role not in user.roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return checker
