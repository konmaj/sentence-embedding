FROM python:3

COPY requirements.txt /tmp/

RUN pip3 install -r /tmp/requirements.txt

ENTRYPOINT ["python3", "-u"]
