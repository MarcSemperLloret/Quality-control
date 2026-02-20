FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "scripts/run_experiment.py", "--input", "avamet_all.parquet", "--workdir", "."]

