FROM python:3.7

ADD .  /flask_app/

WORKDIR /flask_app

RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    python-dev libxml2-dev libxslt1-dev antiword poppler-utils \
    python-pip zlib1g-dev

RUN pip install -r ./requirements.txt

EXPOSE 3030

WORKDIR /flask_app/

CMD ["python","app/index.py"]
