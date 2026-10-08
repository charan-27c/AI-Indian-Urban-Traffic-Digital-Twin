from typing import Dict

from src.perception.junction.junction_config import (
    JunctionConfig,
)

from src.perception.junction.topology import (
    Approach,
    JunctionTopology,
    Lane,
)

from src.perception.junction.topology_provider import (
    TopologyProvider,
)


class ConfigTopologyProvider(TopologyProvider):
    """
    Temporary topology provider.

    Uses junctions.yaml while the actual road-network
    topology is not available.

    This is intentionally limited to junction/approach
    information.

    Lanes and movements are NOT hard-coded here.
    """

    def __init__(
        self,
        config_path: str,
    ):
        self.config = JunctionConfig(
            config_path
        )

    def discover_junction(
        self,
        junction_id: str,
    ) -> JunctionTopology:

        junction = self.config.get_junction(
            junction_id
        )

        topology = JunctionTopology(
            junction_id=junction.junction_id
        )

        for approach_name in (
            junction.get_approaches()
        ):

            approach = Approach(
                approach_id=approach_name,
                name=approach_name,
            )

            topology.add_approach(
                approach
            )

        return topology

    def discover_all_junctions(
        self,
    ) -> Dict[str, JunctionTopology]:

        result = {}

        for junction_id in (
            self.config.get_all_junctions()
        ):

            result[junction_id] = (
                self.discover_junction(
                    junction_id
                )
            )

        return result