FROM python
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt


RUN mkdir /app
RUN mkdir /app/dcoker
RUN mkdir /app/pipeline

COPY docker /app/docker
COPY pipeline /app/pipeline
COPY . /app


COPY pipeline/model/house_price.py /app/pipeline/model/house_price.py
COPY app.py /app/app.py

WORKDIR /app

CMD ["python","app.py"]