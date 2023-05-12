FROM --platform=linux/amd64 python:3.9-slim-buster

ENV AzureWebJobsScriptRoot=/app \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app

CMD [ "func", "host", "start" ]
