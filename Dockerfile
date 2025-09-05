# Stage 1: Build a larger image to compile dlib
FROM python:3.10 as builder

# Install system dependencies for dlib, opencv, and sounddevice
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
    libavutil-dev \
    libswscale-dev \
    libasound2-dev \
    libgl1 \
    libportaudio2 \
    portaudio19-dev

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create a smaller, final image
FROM python:3.10-slim

# Install final runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libportaudio2 \
    portaudio19-dev

# Set the working directory
WORKDIR /app

# Copy Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy the application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to start your Streamlit app
CMD ["streamlit", "run", "app.py"]
