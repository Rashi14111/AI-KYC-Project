# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies for the required libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavdevice-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libasound2-dev \
    libgl1 \
    libportaudio2 \
    portaudio19-dev

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to start your Streamlit app
CMD ["streamlit", "run", "app.py"]
