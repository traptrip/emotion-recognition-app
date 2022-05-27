import os
from datetime import datetime, timedelta
from typing import List

from fastapi import Depends, File, FastAPI, HTTPException, status, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from . import crud, models, schemas
from .database import SessionLocal, engine

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "bbc2b77d1e47c66d05e92ea46f981f67bca330c6b006beeb93fb2a7145bb3ade"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 365

models.Base.metadata.create_all(bind=engine)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# AUTH UTILS
def _create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def _get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user


async def _get_current_active_user(
    current_user: models.User = Depends(_get_current_user),
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    access_token = _create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="This username already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get metadata for all users"""
    if current_user:
        users = crud.get_users(db, skip=skip, limit=limit)
        return users
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorised!",
    )


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get current user metadata"""
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.id == current_user.id:
        return db_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Forbidden!",
    )


@app.post("/tasks/", response_model=schemas.Task)
async def create_task_for_user(
    file: UploadFile = File(),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Post new task for current user"""
    video_name = file.filename
    video_bytes = await file.read()
    return crud.create_user_task(
        db=db, video_name=video_name, video_bytes=video_bytes, user_id=current_user.id
    )


@app.get("/tasks/", response_model=List[schemas.Task])
def read_tasks(
    skip: int = 0,
    limit: int = 1000,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get all tasks for current user"""
    tasks = crud.get_user_tasks(db, current_user.id, skip=skip, limit=limit)
    return tasks


@app.get("/tasks/{task_id}", response_model=schemas.Task)
def read_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get task metadata"""

    task = crud.get_user_task(db, current_user.id, task_id)
    return task


@app.get("/tasks/{task_id}/video")
def read_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get Task video results"""

    task = crud.get_user_task(db, current_user.id, task_id)
    video_path = task.result_video_url
    if os.path.exists(video_path):
        return FileResponse(video_path)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found!")


@app.get("/tasks/{task_id}/meta")
def read_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Get Task table results"""

    task = crud.get_user_task(db, current_user.id, task_id)
    table_path = task.result_table_url
    if os.path.exists(table_path):
        return FileResponse(table_path)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found!")


@app.delete("/tasks/")
def delete_task_for_user(
    task_id: int,
    skip: int = 0,
    limit: int = 1000,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(_get_current_active_user),
):
    """Delete task"""
    result = crud.delete_user_task(db, current_user.id, task_id, skip=skip, limit=limit)
    if result["status"] == "OK":
        return status.HTTP_202_ACCEPTED
    return HTTPException(status_code=404, detail="Task not found")
