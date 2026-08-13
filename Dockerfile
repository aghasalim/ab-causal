FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY src/ src/
COPY app/ app/

# Fetch data and generate the result tables at build time so a cold container
# has the Evidence tab populated on first load rather than showing an empty
# state until someone runs a make target.
RUN python -m src.abcausal.data \
 && python -m src.abcausal.experiments.peeking \
 && python -m src.abcausal.experiments.cuped_gain \
 && python -m src.abcausal.experiments.lalonde

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
