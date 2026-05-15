# Optional AWS cleanup (after removing edge from this repo)

**Your Raspberry Pi is unaffected** by deleting files in GitHub: nothing on the SD card changes until you run commands *on the Pi* or deploy from AWS again.

If you no longer use edge inference from this account, you can **manually** remove leftover cloud resources to avoid confusion or small ongoing costs. Examples (adjust region and names):

```bash
export AWS_REGION=eu-central-1

# List ECR repos you might delete if unused
aws ecr describe-repositories --region $AWS_REGION --query 'repositories[].repositoryName' --output table

# Example: delete old edge repos (only if you are sure)
# aws ecr delete-repository --repository-name robops/ros2-full-stack --force --region $AWS_REGION
# aws ecr delete-repository --repository-name robops/inference --force --region $AWS_REGION
# aws ecr delete-repository --repository-name robops/ros2-stack --force --region $AWS_REGION

# Greengrass: remove deployments / components in AWS IoT console, or use CLI
# aws greengrassv2 list-deployments --region $AWS_REGION
```

Keep **`robops/training`** if you still use CI to build the amd64 training image.

GitHub: you may remove unused secrets (`PI_HOST`, SSH keys for Pi deploy) from **Repository → Settings → Secrets**.
