# ClimateLens Azure ML Infrastructure

## Metadata

| Field | Value |
|------|------|
| Workspace Name | nlp_llm_mvp1 |
| Workspace Owner(s) | Helena Yu |
| Last Updated | Feb 12, 2026 by Karim |

## Compute Instance Configuration

Virtual Machine/Compute Instance Name:  
VM Size/Series:  

## Data Download Process

```
cd [location to folder]
zip -r [name your zip folder].zip [folder name]/
```

## Custom Kernel Setup

Dependency Management:

Dependencies are managed via requirements.txt or environment.yml.

```
pip install -r requirements.txt
```

## Troubleshooting & Common Issues

Azure can be a little funky sometimes, so you may need to refresh the page/workspace for the changes to show or fixes to take place

### Permission Denied on Storage Account

Ensure correct Azure AD role or SAS token.  
Verify storage keys are correctly configured.

### Kernel Not Showing in Jupyter

Re-run `python -m ipykernel install --user --name=[kernel-name].`  
Restart Jupyter server.

### Package Installation Failures

Use `pip install --upgrade pip setuptools wheel.`  
Verify version compatibility in requirements.txt.

## Support:

Point of Contact:  

https://learn.microsoft.com/en-us/answers/questions/5734950/why-is-my-ml-azure-compute-instance-is-always-sett

https://azure.microsoft.com/en-us/pricing/details/storage/blobs/

https://learn.microsoft.com/en-us/answers/questions/2200654/how-do-i-configure-my-azure-ml-workspace-such-that