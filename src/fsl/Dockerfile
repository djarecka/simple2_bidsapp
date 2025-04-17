FROM vnmd/fsl_6.0.7.16

RUN mkdir -p /opt/fsl_wrapper

COPY . /opt/fsl_wrapper

RUN pip install --no-cache-dir -r /opt/fsl_wrapper/requirements.txt

WORKDIR /opt/fsl_wrapper

ENTRYPOINT ["python", "/opt/fsl_wrapper/run.py"]
