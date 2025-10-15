import logging
from intersect_sdk import (
    HierarchyConfig,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
)
from capability import ActiveLearningCapability

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    config = IntersectServiceConfig(
        hierarchy=HierarchyConfig(
            organization='my-org',
            facility='my-facility',
            system='active-learning',
            service='neutron-phasing-service',
        ),
        brokers=[{
            'username': 'intersect_username',
            'password': 'intersect_password',
            'port': 1883,
            'protocol': 'mqtt3.1.1',
        }],
    )

    capability = ActiveLearningCapability()
    service = IntersectService([capability], config)

    logging.info("Starting Active Learning Service. Use Ctrl+C to exit.")
    default_intersect_lifecycle_loop(service)
