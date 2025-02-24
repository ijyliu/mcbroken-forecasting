
# Use the AWS Lambda base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Install Mamba (faster dependency solver)
RUN conda install -c conda-forge mamba -y && conda clean -a

# Copy environment.yaml and create environment using Mamba
COPY environment.yaml /tmp/environment.yaml
RUN mamba env create -f /tmp/environment.yaml

# Activate environment and set PATH
SHELL ["bash", "-c"]
ENV PATH=/opt/conda/envs/mcbroken-forecasting/bin:$PATH
RUN echo "source activate mcbroken-forecasting" > ~/.bashrc

