FROM nvcr.io/nvidia/tensorflow:23.03-tf2-py3

# Install Python dependencies
RUN pip install tensorflow-probability==0.19.0 \
    seaborn \
    codecarbon==2.3.1 \
    causal-learn \
    keras-tuner==1.1.3 \
    tf2onnx==1.14.0 \
    onnxruntime==1.15.1 \
    pandas \
    numpy \
    scikit-learn \
    matplotlib

# Install system dependencies
RUN apt-get update && apt-get -y install tmux
