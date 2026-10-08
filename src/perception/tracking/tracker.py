from ultralytics import YOLO


class VehicleTracker:


    def __init__(
        self,
        model_name="yolo11n.pt"
    ):

        self.model = YOLO(
            model_name
        )


        self.track_history = {}



    def track(
        self,
        frame
    ):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )


        vehicles = []


        for result in results:


            if result.boxes.id is None:
                continue



            boxes = result.boxes


            ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )


            classes = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )


            confs = (
                boxes.conf
                .cpu()
                .numpy()
            )


            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )



            for i, vehicle_id in enumerate(ids):


                x1,y1,x2,y2 = (
                    xyxy[i]
                    .astype(int)
                )


                vehicles.append(
                    {

                    "vehicle_id":
                        int(vehicle_id),

                    "class":
                        self.model.names[
                            classes[i]
                        ],

                    "confidence":
                        float(confs[i]),

                    "bbox":
                        [
                         int(x1),
                         int(y1),
                         int(x2),
                         int(y2)
                        ]

                    }
                )


        return vehicles