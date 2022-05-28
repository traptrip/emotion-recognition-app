from typing import List, Optional
from pydantic import BaseModel
from fastapi import Form

# TASK
class TaskBase(BaseModel):
    pass


class Task(TaskBase):
    id: str
    owner_id: int
    video_path: str
    status: Optional[str]
    result_video_url: Optional[str]
    result_table_url: Optional[str]

    class Config:
        orm_mode = True


# USER
class UserBase(BaseModel):
    username: str = Form()


class UserCreate(UserBase):
    password: str = Form()


class User(UserBase):
    id: int
    is_active: bool
    tasks: List[Task] = []

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str
