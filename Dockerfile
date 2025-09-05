# Use a slim Python image to keep the size small
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libportaudio2 \
    portaudio19-dev \
    cmake

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to start your Streamlit app
CMD ["streamlit", "run", "app.py"]
