FROM python:3.12-slim as builder

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade \
    pip poetry

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1  
   
CMD streamlit run ./app.py