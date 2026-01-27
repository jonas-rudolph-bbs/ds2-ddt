#!/bin/bash

# Variables
IMAGE_NAME="data-quality"
CONTAINER_NAME="data-quality"
DOCKERFILE_PATH="./Dockerfile"  # Update the path to your Dockerfile if needed

# Check if the container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' exists. Stopping and removing it..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
else
    echo "No existing container named '${CONTAINER_NAME}' found."
fi

# Check if the image exists
if docker images --format '{{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
    echo "Image '${IMAGE_NAME}' exists. Removing it..."
    docker rmi ${IMAGE_NAME}
else
    echo "No existing image named '${IMAGE_NAME}' found."
fi

# Build the image
echo "Building the image '${IMAGE_NAME}'..."
docker build -t ${IMAGE_NAME} -f ${DOCKERFILE_PATH} .

# Run the container
echo "Running a new container from the image '${IMAGE_NAME}'..."
docker run --name ${CONTAINER_NAME} -d ${IMAGE_NAME}

echo "Process complete."
