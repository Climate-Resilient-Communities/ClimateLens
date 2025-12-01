from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from datetime import datetime

ml = MLClient.from_config(credential=DefaultAzureCredential())

env = Environment(
    name="bertopic-py310",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

job = command(
    code=".",
    command="bash azureml/run_scripts_test.sh",
    compute="cpu-cluster",
    environment=env,
    environment_variables={
    "CODE_DIR": "./code"
},
    experiment_name="climatelens-script-run",
    display_name=f"climatelens-{datetime.utcnow().strftime('%H%M%S')}",
)

submitted = ml.jobs.create_or_update(job)
print("Submitted job:", submitted.name)

ml.jobs.stream(submitted.name)