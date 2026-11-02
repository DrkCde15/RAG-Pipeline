FROM apache/airflow:2.10.5-python3.12

USER airflow
COPY --chown=airflow:root pyproject.toml README.md /opt/rag/
COPY --chown=airflow:root app /opt/rag/app
RUN pip install --no-cache-dir --constraint \
    "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.1+cpu"
RUN pip install --no-cache-dir --constraint \
    "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt" \
    "/opt/rag"
ENV PYTHONPATH="/opt/rag:${PYTHONPATH}"
