# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your app
COPY . /app

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]



