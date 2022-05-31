FROM python:3.8.12-slim-buster

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
