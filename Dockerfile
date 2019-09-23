FROM python:3.6.9
MAINTAINER Andreas Vlachos <a.vlachos@sheffield.ac.uk>

RUN apt-get update -y
RUN apt-get install -y git

RUN git clone https://github.com/athnlp/athnlp-labs.git
WORKDIR /athnlp-labs

RUN sh setup_dependencies_Docker.sh
