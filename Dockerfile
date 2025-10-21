# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/*.py src/
COPY *.py .

# Install dependencies
RUN pip install -r requirements.txt

# Expose as a CLI
ENTRYPOINT ["python", "main.py"]

# docker build -t convert-to-ome .
# docker build -t cellularimagingcf/convert-to-ome:v0.0.1 .

# WSL Example usage:

# sudo mkdir -p /mnt/data
# sudo mount -t drvfs L:/Archief/active/cellular_imaging/OMERO_test/ValidateDocker /mnt/data

# docker run --rm -v "D:\slides\DB:/data" convert-to-ome --inputfile /data/TestData1/experiment.db --output_folder "/data" --show_progress
