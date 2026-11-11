# Container for the artifact: Python 3.11.15 on Debian bookworm, packages pinned in requirements.lock.txt.
FROM python:3.11.15-slim-bookworm@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/*
WORKDIR /artifact
COPY environment/requirements.lock.txt environment/requirements.lock.txt
RUN pip install --no-cache-dir -r environment/requirements.lock.txt
COPY . .
CMD ["make", "check"]
