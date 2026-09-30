# The triage gear's image. Build from the repo root, so the gear carries the same oct_checks/ code the
# analysis ran:
#   docker build -f flywheel_gears/triage/Dockerfile -t local/triage:0.2.1 .
FROM python:3.12-slim

# Flywheel's gear contract: inputs, outputs and config.json are mounted under /flywheel/v0.
ENV FLYWHEEL=/flywheel/v0
WORKDIR ${FLYWHEEL}

# Dependencies first (cached layer), then the code.
COPY flywheel_gears/triage/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY oct_checks/ oct_checks/
COPY flywheel_gears/triage/manifest.json flywheel_gears/triage/run.py flywheel_gears/triage/triage.py flywheel_gears/triage/oct_layers.py ./
