# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install necessary build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

ENV CHROMA_DB_HOST="18.195.88.154"
ENV CHROMA_DB_PORT="8000"

# Expose port 80 for Streamlit
EXPOSE 80

# Run app.py when the container launches
CMD ["streamlit", "run", "nl2sql/nl2sql.py"]
