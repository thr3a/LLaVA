FROM --platform=linux/x86_64 thr3a/cuda12.1-torch:latest

RUN pip install git+https://github.com/thr3a/LLaVA.git
RUN pip install flash-attn --no-build-isolation
