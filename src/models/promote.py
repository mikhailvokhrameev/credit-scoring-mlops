"""Promotion gate: move the `champion` alias to the `challenger` version if it passes the checks."""
import argparse
import logging
import math
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

from src.config import load_params
from src.models.train import CHALLENGER_ALIAS, MODEL_REGISTRY

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"
AUC_METRIC = "holdout_roc_auc"
DI_METRIC = "holdout_fairness_min_di"


# MLflow 3.x signals a missing alias with INVALID_PARAMETER_VALUE and a missing model with
# RESOURCE_DOES_NOT_EXIST; any other registry error must propagate rather than look like "no champion".
_NOT_FOUND_CODES = {INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST}


def _alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as e:
        if e.error_code in {ErrorCode.Name(c) for c in _NOT_FOUND_CODES}:
            return None
        raise


def _finite_metric(client: MlflowClient, version, metric: str, role: str) -> float | None:
    """Read a metric from the version's run; warn and return None when missing or non-finite."""
    value = client.get_run(version.run_id).data.metrics.get(metric)
    if value is None or not math.isfinite(value):
        logger.warning(f"Not promoting: {role} v{version.version} has missing or non-finite {metric} ({value})")
        return None
    return value


def promote(registered_name: str, params: dict, client: MlflowClient | None = None) -> bool:
    """Set alias `champion` on the `challenger` version when it passes the fairness and AUC gates.

    Fails closed: missing or non-finite metrics on the challenger (or a non-finite champion AUC)
    reject the promotion. Returns True when the challenger is (or already was) the champion.
    """
    client = client or MlflowClient(params["mlflow"]["tracking_uri"])
    challenger = _alias_version(client, registered_name, CHALLENGER_ALIAS)
    if challenger is None:
        logger.warning(f"Not promoting: {registered_name} has no '{CHALLENGER_ALIAS}' alias")
        return False
    champion = _alias_version(client, registered_name, CHAMPION_ALIAS)
    if champion is not None and champion.version == challenger.version:
        logger.info(f"{registered_name} v{challenger.version} is already the {CHAMPION_ALIAS}")
        return True

    auc = _finite_metric(client, challenger, AUC_METRIC, CHALLENGER_ALIAS)
    min_di = _finite_metric(client, challenger, DI_METRIC, CHALLENGER_ALIAS)
    if auc is None or min_di is None:
        return False

    di_min = params["fairness"]["di_min"]
    if min_di < di_min:
        logger.warning(f"Not promoting: {DI_METRIC}={min_di:.3f} is below fairness.di_min={di_min}")
        return False

    if champion is not None:
        champion_auc = _finite_metric(client, champion, AUC_METRIC, CHAMPION_ALIAS)
        if champion_auc is None:
            return False
        epsilon = params["gate"]["auc_epsilon"]
        if auc < champion_auc - epsilon:
            logger.warning(
                f"Not promoting: challenger AUC {auc:.4f} is worse than champion AUC "
                f"{champion_auc:.4f} by more than gate.auc_epsilon={epsilon}"
            )
            return False

    client.set_registered_model_alias(registered_name, CHAMPION_ALIAS, challenger.version)
    logger.info(f"Promoted {registered_name} v{challenger.version} to '{CHAMPION_ALIAS}' (AUC={auc:.4f}, min DI={min_di:.3f})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the challenger model to champion if it passes the gate")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(), help="Model name")
    parser.add_argument("--base", action="store_true", help="Use the base-feature-set registered model")
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    args = parser.parse_args()

    registered_name = f"credit_scoring_{args.model}{'_base' if args.base else ''}"
    if not promote(registered_name, load_params(args.params)):
        sys.exit(1)


if __name__ == "__main__":
    main()
