FROM python:3.7

ADD .  /flask_app/

WORKDIR /flask_app

RUN python -m pip install --upgrade pip

RUN pip install -r ./requirements.txt

EXPOSE 3030

WORKDIR /flask_app/app/src/

CMD ["python","index.py"]
