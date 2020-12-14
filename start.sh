#!/bin/bash

echo "Starting docker container..."
docker run -p 8081:80 -p 8080:8080 fc_mean:latest
