#!/bin/bash
docker run -p 8501:8501  --mount type=bind,source=$PWD/tf_model,target=/models/lstm_multistep -e MODEL_NAME=lstm_multistep -t tensorflow/serving
