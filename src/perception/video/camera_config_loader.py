from pathlib import Path
from typing import Dict

import yaml

from src.perception.video.camera_config import (
    CameraConfig,
    ROI,
)


class CameraConfigLoader:
    """
    Loads dynamic camera configuration from YAML.

    Camera configuration is independent from the
    VideoInput class.

    The configuration defines:

        camera
        source
        junction
        approach
        ROI
    """

    def __init__(
        self,
        config_path: str,
    ):

        self.config_path = Path(
            config_path
        )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Camera configuration not found: "
                f"{self.config_path}"
            )

        self.cameras: Dict[
            str,
            CameraConfig
        ] = {}

        self._load()

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    def _load(self):

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = yaml.safe_load(
                file
            )

        if not data:
            raise ValueError(
                "Camera configuration is empty."
            )

        camera_data = data.get(
            "cameras",
            {}
        )

        if not isinstance(
            camera_data,
            dict,
        ):
            raise ValueError(
                "'cameras' must be a mapping."
            )

        for camera_id, config in (
            camera_data.items()
        ):

            if not isinstance(
                config,
                dict,
            ):
                raise ValueError(
                    f"Invalid configuration "
                    f"for camera '{camera_id}'."
                )

            source = config.get(
                "source"
            )

            if not source:
                raise ValueError(
                    f"Camera '{camera_id}' "
                    f"has no source."
                )

            camera = CameraConfig(
                camera_id=camera_id,
                source=str(source),
                junction_id=config.get(
                    "junction_id"
                ),
                approach_id=config.get(
                    "approach_id"
                ),
            )

            # ---------------------------------------------
            # ROIs
            # ---------------------------------------------

            rois = config.get(
                "rois",
                []
            )

            if not isinstance(
                rois,
                list,
            ):
                raise ValueError(
                    f"ROIs for camera "
                    f"'{camera_id}' must be a list."
                )

            for roi_data in rois:

                roi = self._parse_roi(
                    camera_id,
                    roi_data,
                )

                camera.add_roi(
                    roi
                )

            self.cameras[
                camera_id
            ] = camera

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    @staticmethod
    def _parse_roi(
        camera_id: str,
        roi_data: dict,
    ) -> ROI:

        if not isinstance(
            roi_data,
            dict,
        ):
            raise ValueError(
                f"Invalid ROI configuration "
                f"for camera '{camera_id}'."
            )

        roi_id = roi_data.get(
            "roi_id"
        )

        if not roi_id:
            raise ValueError(
                f"ROI for camera "
                f"'{camera_id}' has no roi_id."
            )

        points = roi_data.get(
            "points",
            []
        )

        if not isinstance(
            points,
            list,
        ):
            raise ValueError(
                f"ROI '{roi_id}' points "
                f"must be a list."
            )

        parsed_points = []

        for point in points:

            if (
                not isinstance(
                    point,
                    list,
                )
                or len(point) != 2
            ):
                raise ValueError(
                    f"Invalid point in ROI "
                    f"'{roi_id}'."
                )

            x = int(point[0])
            y = int(point[1])

            parsed_points.append(
                (x, y)
            )

        return ROI(
            roi_id=roi_id,
            points=parsed_points,
        )

    # -----------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------

    def get_all_cameras(
        self,
    ) -> Dict[str, CameraConfig]:

        return dict(
            self.cameras
        )

    def get_camera(
        self,
        camera_id: str,
    ) -> CameraConfig:

        if camera_id not in self.cameras:
            raise KeyError(
                f"Camera '{camera_id}' "
                f"not found."
            )

        return self.cameras[
            camera_id
        ]

    def get_cameras_for_junction(
        self,
        junction_id: str,
    ) -> Dict[str, CameraConfig]:

        return {
            camera_id: camera
            for camera_id, camera
            in self.cameras.items()
            if camera.junction_id
            == junction_id
        }

    def get_cameras_for_approach(
        self,
        junction_id: str,
        approach_id: str,
    ) -> Dict[str, CameraConfig]:

        return {
            camera_id: camera
            for camera_id, camera
            in self.cameras.items()
            if (
                camera.junction_id
                == junction_id
                and
                camera.approach_id
                == approach_id
            )
        }