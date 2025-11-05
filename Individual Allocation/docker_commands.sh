docker build -< Dockerfile -t tf-gpu:new1
docker run --gpus all --rm -it -d --ulimit stack=67108864  -v ${PWD}:/workspace tf-gpu:new1