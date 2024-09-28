# Use the official Python base image
FROM python:3.11-slim


# Set the working directory in the container
WORKDIR /app

# Copy the rest of the project
COPY . .

# Install Python dependencies via Poetry
RUN pip install -r requirements.txt

# Default command to run when the container starts
CMD ["python", "challenge_apollo/main.py"]
