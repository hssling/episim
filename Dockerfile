FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY episim /app/episim
COPY apps/hf_space /app/apps/hf_space

RUN pip install --no-cache-dir -e ".[app]"

EXPOSE 7860
CMD ["python", "apps/hf_space/app.py"]
