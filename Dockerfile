FROM python:3.10-slim

WORKDIR /api

COPY ./api/requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

COPY ./api /api

CMD ["fastapi", "run", "/api/main.py", "--port", "80"]
