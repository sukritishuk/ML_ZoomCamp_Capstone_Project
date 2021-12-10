FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
 
RUN pipenv install --system --deploy

COPY ["Capstone_Final_Model_predict.py", "model.bin", "./"]

EXPOSE 5000

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:5000", "Capstone_Final_Model_predict:app"]