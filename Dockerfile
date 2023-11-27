# Use an official Cuda 12.1 runtime as a parent image
FROM nvidia/cuda:12.1-base

RUN apt-get update \
  && apt-get install -y python3-pip python3.10 \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /gorilla-lora

# Copy package.json and package-lock.json
COPY requirements.txt /gorilla-lora/

# Create folders for data
RUN mkdir /gorilla-lora/data/llama-model/7B \
  && mkdir /gorilla-lora/lora/data/api \
  && mkdir /gorilla-lora/lora/data/inst \
  && mkdir /gorilla-lora/lora/data/error \
  && mkdir /gorilla-lora/adapter/checkpoint/ \
  && mkdir /gorilla-lora/lora/results/

# Install any needed packages
RUN pip3 install -r requirements.txt

# Bundle app source inside the Docker image
COPY . .

# Make port available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV LLAMA_PATH="/gorilla-lora/data/llama-model/7B"
ENV DATA_FOLDER="/gorilla-lora/lora/data/api"
ENV OUTPUT_FOLDER="/gorilla-lora/lora/data/inst"
ENV ERROR_FOLDER="/gorilla-lora/lora/data/error"
ENV CKPT_PATH="/gorilla-lora/adapter/checkpoint/"
ENV RESULT_PATH = "/gorilla-lora/lora/results/"

# Run app when the container launches
CMD ["npm", "start"]
