FROM python:3

COPY requirements.txt /tmp/

RUN pip3 install -r /tmp/requirements.txt

RUN ["python3", "-c", "import nltk; nltk.download('punkt')"]

ENTRYPOINT ["python3", "-u"]
