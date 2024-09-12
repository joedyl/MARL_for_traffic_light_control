FROM --platform=linux/amd64 python:3.8

# Upgrade pip, setuptools, and wheel first
RUN pip install --upgrade pip setuptools wheel

# Install other dependencies from requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt