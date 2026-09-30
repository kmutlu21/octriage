# The relay gear's image: Python + flywheel-gear-toolkit + two scripts. No torch, no model, no GPU:
# the model runs in the service (gpu_service/Dockerfile), so this image stays small.
#   docker build -t local/relay:0.2.1 flywheel_gears/relay/      (or: flyw gear build flywheel_gears/relay/)
FROM python:3.12-slim

# Flywheel's gear contract: the engine mounts input/, output/, work/ and config.json under
# /flywheel/v0 and runs the manifest's "command" there, so the scripts must live in that folder.
ENV FLYWHEEL=/flywheel/v0
WORKDIR ${FLYWHEEL}

# Dependencies first, code second: editing run.py then reuses the cached pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manifest.json run.py relay.py ./
