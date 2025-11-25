# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Render assigns a dynamic port (usually 10000), so we don't hardcode it.
# We use the shell form of CMD to allow the $PORT variable to expand.
CMD gunicorn --bind 0.0.0.0:$PORT app:app
