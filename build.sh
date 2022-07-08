#!/bin/bash

echo "Building docker image..."
docker build . --tag featurecloud.ai/fc_nelson_aalen
