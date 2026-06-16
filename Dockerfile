# SPDX-FileCopyrightText: 2025 - 2025 Software GmbH, Darmstadt, Germany and/or its subsidiaries and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Use an official Ubuntu image as a base
FROM python:3.12-slim

# Install system tools, Python, and pip
RUN apt-get update && apt-get install -y \
    curl \
    mosquitto-clients \
    ca-certificates \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Python requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the MQTT ports
# EXPOSE 1883 8883
EXPOSE 8000 8000

# Ensure the Python interpreter is specified explicitly
ENV PYTHONUNBUFFERED=1

# Default command to run the application
CMD ["python3", "main.py"]