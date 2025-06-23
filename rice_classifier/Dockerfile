# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all files from the rice_classifier directory
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port (Render uses 10000+)
EXPOSE 8000

# Start the app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
