FROM hccz95/hfo:latest

RUN pip install --no-cache-dir tensorflow==1.13.1 torch==1.3.0 matplotlib protobuf==3.20.0
