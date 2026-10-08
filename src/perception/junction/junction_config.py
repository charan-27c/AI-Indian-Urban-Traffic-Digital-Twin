from pathlib import Path
from typing import Dict

import yaml

from src.perception.junction.junction_model import Junction


class JunctionConfig:
    """
    Loads junction configuration from YAML.

    This class does not assume that every junction has
    the same number of approaches.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Junction configuration not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)

        if not config:
            raise ValueError(
                "Junction configuration is empty."
            )

        self.junctions: Dict[str, Junction] = {}

        self._load_junctions(
            config.get("junctions", {})
        )

    def _load_junctions(self, junction_data: dict):
        """Create Junction objects from configuration."""

        for junction_id, data in junction_data.items():

            name = data.get(
                "name",
                junction_id
            )

            approaches = data.get(
                "approaches",
                []
            )

            junction = Junction(
                junction_id=junction_id,
                name=name,
                approaches=approaches,
            )

            self.junctions[junction_id] = junction

    def get_junction(
        self,
        junction_id: str
    ) -> Junction:
        """Return one junction."""

        if junction_id not in self.junctions:
            raise KeyError(
                f"Junction '{junction_id}' "
                f"not found."
            )

        return self.junctions[junction_id]

    def get_all_junctions(
        self
    ) -> Dict[str, Junction]:
        """Return all configured junctions."""

        return dict(self.junctions)

    def get_junction_count(self) -> int:
        """Return number of junctions."""

        return len(self.junctions)