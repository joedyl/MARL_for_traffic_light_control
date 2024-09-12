FROM --platform=linux/amd64 python:3.8

# Upgrade pip, setuptools, and wheel first
RUN pip install --upgrade pip setuptools wheel

# Clone and install TinyScaler with modifications
# RUN git clone https://github.com/Farama-Foundation/TinyScaler.git --branch v1.2.6 && \
#     cd TinyScaler && \
#     sed -i 's/extra_compile_args/#extra_compile_args/g' setup.py && \
#     sed -i 's/extra_link_args/#extra_link_args/g' setup.py && \
#     pip install -e . && \
#     cd .. && \
#     rm -rf TinyScaler

# Install other dependencies from requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
    