FROM python:3.10-slim

COPY code_agent.py /code_agent.py
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

ENTRYPOINT ["python", "/code_agent.py"]