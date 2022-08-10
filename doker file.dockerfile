FROM python:3.7

RUN pip install fastapi uvicorn nltk joblib re

COPY ./app /app/app

ENV PYTHONPATH=/app
WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]