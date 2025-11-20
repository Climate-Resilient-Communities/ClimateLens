from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from datetime import datetime

ml = MLClient.from_config(credential=DefaultAzureCredential())

env = Environment(
    name="climatelens-env",
    conda_file="environment.yml",   # <- your environment
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

job = command(
    code=".", # folder with scripts
    command="bash run_all.sh",
    compute=".", #cluster name, good to hide
    environment=env,
    environment_variables={
        "CODE_DIR": ""
    },
    experiment_name="climatelens-script-run",
    display_name=f"climatelens-script-{datetime.utcnow().strftime('%H%M%S')}",
)

submitted = ml.create_or_update(job)
print("Submitted job:", submitted.name)

ml.jobs.stream(submitted.name) #check if can be logged in same log file as others