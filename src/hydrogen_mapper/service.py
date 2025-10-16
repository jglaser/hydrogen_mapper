import argparse
import logging
import pathlib

import yaml
from intersect_sdk import (
    HierarchyConfig,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
)
from hydrogen_mapper.capability import ActiveLearningCapability

def main() -> None:
    # CLI
    parser = argparse.ArgumentParser(
        description="Hydrogen Mapper INTERSECT service"
    )
    parser.add_argument(
        '--config',
        type=pathlib.Path,
        default="config.yaml",
    )
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Get config to setup INTERSECT client
    with open(args.config) as config_reader:
        config_raw = yaml.safe_load(config_reader)

    # Define the service
    print(config_raw)
    config = IntersectServiceConfig(
        **config_raw["intersect"],
    )

    capability = ActiveLearningCapability()
    service = IntersectService([capability], config)

    logging.info("Starting Active Learning Service. Use Ctrl+C to exit.")
    default_intersect_lifecycle_loop(service)

if __name__ == "__main__":
    main()
