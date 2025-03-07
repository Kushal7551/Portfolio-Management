# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Ensure the evaluation script is executable
RUN chmod +x inference.py

# Expose the input and output directories for the user to bind
VOLUME ["/input", "/output"]

EXPOSE 4000

# Define the command to run the script
ENTRYPOINT ["python", "evaluate_model.py"]