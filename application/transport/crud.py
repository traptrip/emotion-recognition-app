"""Create, Read, Update, Delete"""
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from celery.contrib.abortable import AbortableAsyncResult

from . import models, schemas, worker
from .settings import *

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def _get_password_hash(password):
    return pwd_context.hash(password)


# USER
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = _get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not (user and _verify_password(password, user.hashed_password)):
        return False
    return user


# TASK
def _update_task(db: Session, db_task: models.Task, celery_task: AbortableAsyncResult):
    task_status = (
        celery_task.status
        if db_task.status not in ("SUCCESS", "FAILURE")
        else db_task.status
    )
    db_task.status = task_status
    if task_status == "SUCCESS":
        db_task.result_video_url = celery_task.result[0]
        db_task.result_meta_url = celery_task.result[1]
    db.commit()
    return db_task


def get_user_task(db: Session, user_id: int, task_id: str):
    celery_task = AbortableAsyncResult(task_id)
    db_task = db.get(models.Task, task_id)
    if db_task:
        if db_task.owner_id == user_id:
            db_task = _update_task(db, db_task, celery_task)
            return db_task
    return None


def get_user_tasks(db: Session, user_id: int, skip: int = 0, limit: int = 1000):
    db_tasks = (
        db.query(models.Task)
        .filter(models.Task.owner_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    for db_task in db_tasks:
        celery_task = AbortableAsyncResult(db_task.id)
        _update_task(db, db_task, celery_task)
    return db_tasks


def create_user_task(db: Session, video_name: str, video_bytes: bytes, user_id: int):
    user_dir = MEDIA_ROOT / str(user_id)
    user_dir.mkdir(exist_ok=True)
    video_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_filename_{video_name}"
    video_path = str(MEDIA_ROOT / str(user_id) / video_name)
    with open(video_path, "wb") as fout:
        fout.write(video_bytes)

    celery_task = worker.create_task.s().delay(video_path)

    db_task = models.Task(
        id=celery_task.id,
        status=celery_task.status,
        video_path=video_path,
        owner_id=user_id,
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def delete_user_task(db: Session, user_id: int, task_id: int):
    task = db.get(models.Task, task_id)
    if task.owner_id == user_id:
        celery_task = AbortableAsyncResult(task_id)
        while not celery_task.is_aborted():
            celery_task.abort()

        if task.result_video_url:
            os.remove(task.result_video_url)
        if task.result_meta_url:
            os.remove(task.result_meta_url)
        db.delete(task)
        db.commit()
        return {"status": "OK"}
    return {"status": "404"}
