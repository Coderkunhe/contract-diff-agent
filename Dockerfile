FROM python:3.12-slim

# System deps: CJK fonts for PDF export
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cache layer)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[web]"

# App code
COPY web/ web/

RUN mkdir -p /app/data/jobs /app/data/uploads

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
