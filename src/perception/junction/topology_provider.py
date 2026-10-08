from abc import ABC, abstractmethod
from typing import Dict

from src.perception.junction.topology import (
    JunctionTopology,
)


class TopologyProvider(ABC):
    """
    Generic interface for obtaining junction topology.

    The rest of the application should depend on this
    interface instead of directly depending on SUMO.

    Possible future providers:

        ConfigurationTopologyProvider
        SumoTopologyProvider
        OpenStreetMapTopologyProvider
        RealRoadNetworkProvider
    """

    @abstractmethod
    def discover_junction(
        self,
        junction_id: str,
    ) -> JunctionTopology:
        """
        Discover one junction.
        """
        raise NotImplementedError

    @abstractmethod
    def discover_all_junctions(
        self,
    ) -> Dict[str, JunctionTopology]:
        """
        Discover all available junctions.
        """
        raise NotImplementedError