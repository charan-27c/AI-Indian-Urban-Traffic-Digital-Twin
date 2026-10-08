import cv2



def draw_tracks(
        frame,
        vehicles
):


    for vehicle in vehicles:


        x1,y1,x2,y2 = (
            vehicle["bbox"]
        )


        vehicle_id = (
            vehicle["vehicle_id"]
        )


        label = (
            f'ID:{vehicle_id} '
            f'{vehicle["class"]}'
        )


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (255,0,0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,0,0),
            2
        )


    return frame