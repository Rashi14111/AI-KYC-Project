# Use a base image with Python and Ubuntu
FROM python:3.10-slim

# Set a non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install the system dependencies for av, opencv, sounddevice, and dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavdevice-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libasound2-dev \
    libgl1 \
    libportaudio2 \
    portaudio19-dev \
    cmake \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to start your Streamlit app
CMD ["streamlit", "run", "app.py"]
