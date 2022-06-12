import os
import logging
from datetime import timedelta
from typing import List

from fastapi import (
    Depends,
    File,
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    Request,
    Form,
)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "bbc2b77d1e47c66d05e92ea46f981f67bca330c6b006beeb93fb2a7145bb3ade"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 365

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

manager = LoginManager(SECRET_KEY, token_url="/token", use_cookie=True)
manager.cookie_name = "token-cookie"


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# AUTH UTILS
@manager.user_loader
def load_user(username: str):
    with SessionLocal() as db:
        user = crud.get_user_by_username(db, username)
    return user


@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    username = form_data.username
    password = form_data.password
    user = crud.authenticate_user(db, username, password)
    if not user:
        raise InvalidCredentialsException
    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    access_token = manager.create_access_token(
        data={"sub": username}, expires=access_token_expires
    )
    resp = RedirectResponse(url="/main", status_code=status.HTTP_302_FOUND)
    manager.set_cookie(resp, access_token)
    return resp


# User routes
@app.post("/users/add")
def create_user(
    username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)
):
    """Register a new user"""

    db_user = crud.get_user_by_username(db, username=username)
    if db_user:
        raise HTTPException(status_code=400, detail="This username already registered")
    user = schemas.UserCreate(username=username, password=password)
    crud.create_user(db=db, user=user)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.post("/tasks/add")
async def create_task_for_user(
    file: UploadFile = File(),
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Post new task for current user"""

    video_name = file.filename
    video_bytes = await file.read()
    crud.create_user_task(
        db=db, video_name=video_name, video_bytes=video_bytes, user_id=current_user.id
    )

    return RedirectResponse(url="/main", status_code=status.HTTP_302_FOUND)


@app.get("/tasks/", response_model=List[schemas.Task])
def read_tasks(
    skip: int = 0,
    limit: int = 1000,
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Get all tasks for current user"""

    tasks = crud.get_user_tasks(db, current_user.id, skip=skip, limit=limit)
    return tasks


@app.get("/tasks/{task_id}", response_model=schemas.Task)
def read_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Get task metadata"""

    task = crud.get_user_task(db, current_user.id, task_id)
    return task


@app.get("/tasks/{task_id}/video")
def read_task_video(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Get Task video results"""

    task = crud.get_user_task(db, current_user.id, task_id)
    video_path = task.result_video_url
    if os.path.exists(video_path):
        return FileResponse(video_path)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found!")


@app.get("/tasks/{task_id}/meta")
def read_task_meta(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Get Task table results"""

    task = crud.get_user_task(db, current_user.id, task_id)
    table_path = task.result_meta_url
    if os.path.exists(table_path):
        return FileResponse(table_path)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found!")


@app.delete("/tasks/{task_id}")
def delete_task_for_user(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(manager),
):
    """Delete task"""

    result = crud.delete_user_task(db, current_user.id, task_id)
    if result["status"] == "OK":
        return status.HTTP_202_ACCEPTED
    return HTTPException(status_code=404, detail="Task not found")


# Web pages routes
@app.get("/login")
async def login(request: Request):
    """Login"""

    return templates.TemplateResponse("login2.html", context={"request": request})


@app.get("/logout")
async def logout():
    """Logout from current session"""

    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    manager.set_cookie(response, "")
    return response


@app.get("/")
async def index():
    """Startup html page"""
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/main")
async def main_page(request: Request, current_user=Depends(manager)):
    """Main html page"""

    return templates.TemplateResponse(
        "index.html", context={"request": request, "user": current_user}
    )
