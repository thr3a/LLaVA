FROM --platform=linux/x86_64 thr3a/cuda12.1-torch:latest

COPY ./requirements.txt ./
RUN pip install git+https://github.com/haotian-liu/LLaVA.git
RUN pip install flash-attn --no-build-isolation
