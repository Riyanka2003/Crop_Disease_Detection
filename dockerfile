# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port on which the app runs (standard for Hugging Face)
EXPOSE 7860

# Define the command to run your app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]