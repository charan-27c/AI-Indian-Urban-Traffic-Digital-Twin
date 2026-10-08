from typing import Dict, List

from src.perception.video.camera_config import (
    CameraConfig,
)

from src.perception.junction.junction_config import (
    JunctionConfig,
)


class CameraConfigValidator:
    """
    Validates camera configuration against
    the configured junction topology.

    Validation flow:

        Camera
          ↓
        Junction
          ↓
        Approach

    No fixed number of approaches or cameras
    is assumed.
    """

    def __init__(
        self,
        junction_config: JunctionConfig,
    ):
        self.junction_config = junction_config

    # -----------------------------------------------------
    # VALIDATE ALL CAMERAS
    # -----------------------------------------------------

    def validate_all(
        self,
        cameras: Dict[str, CameraConfig],
    ) -> List[str]:
        """
        Validate all configured cameras.

        Returns a list of validation errors.

        Empty list = configuration is valid.
        """

        errors = []

        for camera_id, camera in cameras.items():

            camera_errors = self.validate_camera(
                camera
            )

            errors.extend(
                camera_errors
            )

        return errors

    # -----------------------------------------------------
    # VALIDATE ONE CAMERA
    # -----------------------------------------------------

    def validate_camera(
        self,
        camera: CameraConfig,
    ) -> List[str]:
        """
        Validate one camera.

        Checks:

            1. junction_id exists
            2. approach_id exists
            3. approach belongs to junction
        """

        errors = []

        # -------------------------------------------------
        # CAMERA ID
        # -------------------------------------------------

        if not camera.camera_id:

            errors.append(
                "Camera ID cannot be empty."
            )

            return errors

        # -------------------------------------------------
        # JUNCTION
        # -------------------------------------------------

        if not camera.junction_id:

            errors.append(
                f"Camera '{camera.camera_id}' "
                f"does not have a junction_id."
            )

            return errors

        try:

            junction = (
                self.junction_config.get_junction(
                    camera.junction_id
                )
            )

        except (
            KeyError,
            ValueError,
        ):

            errors.append(
                f"Camera '{camera.camera_id}' "
                f"references unknown junction "
                f"'{camera.junction_id}'."
            )

            return errors

        # -------------------------------------------------
        # APPROACH
        # -------------------------------------------------

        if not camera.approach_id:

            errors.append(
                f"Camera '{camera.camera_id}' "
                f"does not have an approach_id."
            )

            return errors

        # -------------------------------------------------
        # APPROACH BELONGS TO JUNCTION
        # -------------------------------------------------

        approaches = (
            junction.get_approaches()
        )

        if camera.approach_id not in approaches:

            errors.append(
                f"Camera '{camera.camera_id}' "
                f"references approach "
                f"'{camera.approach_id}' "
                f"which does not exist in "
                f"junction '{camera.junction_id}'."
            )

        return errors