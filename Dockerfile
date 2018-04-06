FROM python:3

# Adding Java
RUN echo "deb http://http.debian.net/debian jessie-backports main" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -t jessie-backports  -y openjdk-8-jre

# Stanford CoreNLP configuration
ENV STANFORD_DIR="/opt/resources/stanford"
ENV CLASSPATH=${STANFORD_DIR}/stanford-postagger-full-2015-04-20/stanford-postagger.jar:${STANFORD_DIR}/stanford-ner-2015-04-20/stanford-ner.jar:${STANFORD_DIR}/stanford-parser-full-2015-04-20/stanford-parser.jar:${STANFORD_DIR}/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar
ENV STANFORD_MODELS=$STANFORD_DIR/stanford-postagger-full-2015-04-20/models:$STANFORD_DIR/stanford-ner-2015-04-20/classifiers

# Python libraries
COPY requirements/* /tmp/

RUN pip3 install -r /tmp/1_math.txt

RUN pip3 install -r /tmp/2_nlp.txt
RUN ["python3", "-c", "import nltk; nltk.download('punkt')"]

RUN pip3 install -r /tmp/3_ml.txt
RUN pip3 install -r /tmp/4_other.txt

ENTRYPOINT ["python3", "-u"]
