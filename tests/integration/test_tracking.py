from pathlib import Path

import cv2


from src.perception.video.video_input import VideoInput

from src.perception.tracking.tracker import VehicleTracker

from src.perception.tracking.visualizer import draw_tracks



ROOT = Path(__file__).resolve().parents[2]


VIDEO = (
    ROOT
    /
    "data"
    /
    "video"
    /
    "traffic_test.mp4"
)



def main():


    print(
        "PHASE 3 - VEHICLE TRACKING"
    )


    video = VideoInput(
        str(VIDEO),
        "J1_CCTV"
    )


    video.open()



    tracker = VehicleTracker()



    while True:


        success, frame = video.read()


        if not success:
            break



        vehicles = tracker.track(
            frame
        )


        frame = draw_tracks(
            frame,
            vehicles
        )


        print(
            "Active vehicles:",
            len(vehicles)
        )


        cv2.imshow(
            "ByteTrack",
            frame
        )


        if cv2.waitKey(1)&0xff == ord("q"):
            break



    video.release()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()