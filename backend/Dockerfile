FROM python:3.8.3-slim

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

WORKDIR /app
COPY requirements.txt /app


RUN python3 -m pip install --trusted-host pypi.python.org -r requirements.txt  

COPY . /app
# execute the Flask app
CMD ["python", "app.py"]