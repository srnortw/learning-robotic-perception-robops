"""
Phase E — Promote Canary to Fleet

After the canary Pi passes Phase F monitoring, this script creates a
Greengrass v2 deployment targeting group-fleet with the same component
versions that were deployed to group-canary.

Usage:
    python pipeline/phase_e/promote_to_fleet.py \
        --dataset-version v1 \
        [--region eu-central-1] \
        [--dry-run]
"""

import argparse
import json
import os
import sys

import boto3
import yaml

PARAMS_PATH     = "pipeline/phase_c/detr/params.yaml"
ACCOUNT_ID      = "688567275774"
AWS_REGION      = "us-east-1"   # Greengrass registered in us-east-1
CANARY_GROUP    = "group-canary"
FLEET_GROUP     = "group-fleet"

INFERENCE_COMPONENT  = "com.robops.inference"
ROS2STACK_COMPONENT  = "com.robops.ros2stack"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def version_from_dataset(dataset_version: str) -> str:
    """Convert dataset version string (e.g. 'v1') → Greengrass semver ('1.0.0')."""
    num = dataset_version.lstrip("v")
    return f"{num}.0.0"


def get_canary_deployment(client, canary_arn: str) -> dict | None:
    """Find the latest active deployment for the canary group."""
    paginator = client.get_paginator("list_deployments")
    for page in paginator.paginate(targetArn=canary_arn):
        deployments = page.get("deployments", [])
        active = [d for d in deployments if d.get("deploymentStatus") == "COMPLETED"]
        if active:
            # most recent first
            active.sort(key=lambda d: d.get("creationTimestamp", ""), reverse=True)
            return client.get_deployment(deploymentId=active[0]["deploymentId"])
    return None


def promote(dataset_version: str, region: str, dry_run: bool):
    component_version = version_from_dataset(dataset_version)
    fleet_arn  = f"arn:aws:iot:{region}:{ACCOUNT_ID}:thinggroup/{FLEET_GROUP}"
    canary_arn = f"arn:aws:iot:{region}:{ACCOUNT_ID}:thinggroup/{CANARY_GROUP}"

    client = boto3.client("greengrassv2", region_name=region)

    print(f"Looking up latest canary deployment for {CANARY_GROUP}...")
    canary_deployment = get_canary_deployment(client, canary_arn)

    if canary_deployment:
        # Re-use the exact component versions + configs from canary deployment
        components = canary_deployment.get("components", {})
        print(f"Found canary deployment — re-using {len(components)} components")
    else:
        # Fallback: build from known component versions
        print("No canary deployment found — building component map from dataset version")
        components = {
            INFERENCE_COMPONENT: {
                "componentVersion": component_version,
            },
            ROS2STACK_COMPONENT: {
                "componentVersion": component_version,
            },
        }

    deployment_config = {
        "targetArn": fleet_arn,
        "deploymentName": f"robops-fleet-detr-{dataset_version}",
        "components": components,
        "deploymentPolicies": {
            "failureHandlingPolicy": "ROLLBACK",
            "componentUpdatePolicy": {
                "failureHandlingPolicy": "ROLLBACK",
                "action": "NOTIFY_COMPONENTS",
                "timeoutInSeconds": 60,
            },
            "configurationValidationPolicy": {
                "timeoutInSeconds": 60,
            },
        },
        "iotJobConfiguration": {
            "jobExecutionsRolloutConfig": {
                "exponentialRate": {
                    "baseRatePerMinute": 5,
                    "incrementFactor": 2.0,
                    "rateIncreaseCriteria": {"numberOfSucceededThings": 3},
                },
                "maximumPerMinute": 50,
            },
            "abortConfig": {
                "criteriaList": [
                    {
                        "failureType": "FAILED",
                        "action": "CANCEL",
                        "thresholdPercentage": 20.0,
                        "minNumberOfExecutedThings": 3,
                    }
                ]
            },
        },
    }

    print(f"\nFleet deployment config:")
    print(json.dumps(deployment_config, indent=2, default=str))

    if dry_run:
        print("\n[DRY RUN] Skipping actual Greengrass deployment.")
        return

    response = client.create_deployment(**deployment_config)
    deployment_id = response["deploymentId"]
    print(f"\n✓ Fleet deployment created: {deployment_id}")
    print(f"  Target: {fleet_arn}")
    print(f"  Monitor at: https://{region}.console.aws.amazon.com/iot/home?region={region}#/greengrass/v2/deployments")


def main():
    parser = argparse.ArgumentParser(description="Phase E — promote canary to fleet")
    parser.add_argument("--dataset-version", required=True, help="e.g. v1")
    parser.add_argument("--region",   default=AWS_REGION)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print deployment config without creating it")
    args = parser.parse_args()

    promote(args.dataset_version, args.region, args.dry_run)


if __name__ == "__main__":
    main()
