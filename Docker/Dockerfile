FROM mcr.microsoft.com/azureml/base:openmpi3.1.2-ubuntu16.04
RUN conda install -c r -y r-essentials=3.6.0 r-reticulate rpy2 r-remotes r-e1071 r-optparse && conda clean -ay && pip install --no-cache-dir azureml-defaults && pip install --no-cache-dir azureml-sdk
ENV TAR="/bin/tar"
RUN R -e "remotes::install_github('https://github.com/Azure/azureml-sdk-for-r')"
USER root
RUN mkdir -p $HOME/.cache
WORKDIR /
RUN if [ $SPARK_HOME ]; then /bin/bash -c '$SPARK_HOME/bin/spark-submit  /azureml-environment-setup/spark_cache.py'; fi
ENV AZUREML_ENVIRONMENT_IMAGE True
CMD ["bash"]
RUN R -e "system('chmod 777 /opt/miniconda/lib/R/library')"
CMD ["bash"]