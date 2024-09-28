# Define the image name and service
SERVICE_NAME = app

.PHONY: install run build start stop

# Install dependencies using Poetry (for local development)
install:
	pip install -r requirements.txt

# Run the main Python script (for local development)
run:
	python3 challenge_apollo/main.py

# Build the Docker image using Docker Compose
build:
	docker compose build

# Start the Docker container using Docker Compose
start:
	docker compose up -d

# Stop the Docker container
stop:
	docker compose down

# Run the main Python script inside the Docker container
run-container:
	docker compose run $(SERVICE_NAME)
