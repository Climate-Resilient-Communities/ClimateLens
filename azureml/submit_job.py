from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from datetime import datetime

ml = MLClient.from_config(credential=DefaultAzureCredential())

job = command(
    code=".", # folder with scripts
    command="bash run_all.sh",
    environment="azureml:env-name",#env name, good to hide
    environment=Environment(
        image="..." #env name, good to hide
    ),
    compute=".", #cluster name, good to hide
    environment_variables={
        "CODE_DIR": ""
    },
    experiment_name="multi-script-run",
    display_name=f"multi-script-{datetime.utcnow().strftime('%H%M%S')}",
)

submitted = ml.create_or_update(job)
print("Submitted job:", submitted.name)

ml.jobs.stream(submitted.name) #check if can be logged in same log file as others