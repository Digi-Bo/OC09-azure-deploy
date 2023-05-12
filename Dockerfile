# Use the official Python base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install any needed system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libgomp1 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Azure Functions Core Tools
RUN curl -sL https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb --output packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y azure-functions-core-tools-3

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed Python packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 80

# Start the app
CMD ["func", "start", "--port", "80"]
