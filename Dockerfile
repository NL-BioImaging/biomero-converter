# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE.md ./
COPY src/biomero_converter/*.py src/biomero_converter/

# Version from the GitHub release tag (no .git in the image)
ARG VERSION=unknown
ENV BIOMERO_CONVERTER_VERSION=$VERSION

# Install the package and its dependencies (setuptools-scm needs the version without .git)
RUN version=$(echo "$VERSION" | sed 's/^v//');     case "$version" in [0-9]*) ;; *) version=0.0.0 ;; esac;     SETUPTOOLS_SCM_PRETEND_VERSION=$version pip install --no-cache-dir .

# Expose as a CLI
ENTRYPOINT ["biomero-converter"]

# docker build -t biomero-converter .
# docker build -t cellularimagingcf/biomero-converter:latest .

# WSL Example usage:

# sudo mkdir -p /mnt/data
# sudo mount -t drvfs L:/Archief/active/cellular_imaging/OMERO_test/ValidateDocker /mnt/data

# docker run --rm -v "D:\slides\DB:/data" biomero-converter --inputfile /data/TestData1/experiment.db --outputfolder "/data" --show_progress
