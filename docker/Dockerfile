FROM nvcr.io/nvidia/pytorch:19.12-py3

# update and install setup
RUN apt-get update && apt-get install -y \
        libmysqlclient-dev

# no more need for sudo
#RUN useradd -r -u 12078 mnoukhov
#USER mnoukhov

# pip install
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# nltk data
RUN python -m nltk.downloader wordnet
