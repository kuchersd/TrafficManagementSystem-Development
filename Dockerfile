# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system packages required by OpenGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install system package required by libgthread-2.0.so.0
RUN apt-get update && apt-get install -y libglib2.0-0

RUN apt-get install gcc -y

RUN apt-get -y install git

RUN pip install Cython

RUN pip install numpy
RUN pip install lap==0.4.0
# Install lap
#RUN pip install numpy
#RUN git clone https://github.com/gatagat/lap.git
#RUN pip install -e lap/

# Set an environment variable to ensure that OpenGL runs in headless mode
ENV MESA_GLSL_CACHE_DISABLE=true

# Install any necessary dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port number that the FastAPI server will listen on
EXPOSE 8000

# Define the command to run when the container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
