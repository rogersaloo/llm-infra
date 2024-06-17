FROM python:3.9-slim-buster

WORKDIR /app
ADD . /app

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", ":8090:80"]