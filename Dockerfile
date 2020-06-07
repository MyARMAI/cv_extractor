FROM python:3.7

ADD .  /flask_app/

WORKDIR /flask_app

RUN pip install -r ./requirements.txt

EXPOSE 3030

WORKDIR /flask_app/app/src/

CMD ["python","index.py"]
