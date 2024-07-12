# app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    # build-essential \
    # software-properties-common \
    # git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "The_Generator.py", "--server.port=8501", "--server.address=0.0.0.0"]