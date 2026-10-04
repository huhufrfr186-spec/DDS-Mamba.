FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir numpy==1.26.4 Pillow==10.3.0 && python -m pip install -e . --no-deps
# Optional after build: pip install mamba-ssm==2.2.2 --no-build-isolation
CMD ["python", "-m", "dds_mamba", "--help"]
