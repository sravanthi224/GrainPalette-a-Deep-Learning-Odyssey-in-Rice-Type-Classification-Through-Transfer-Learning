# Use a base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy files from host into container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 10000

# Start the app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
