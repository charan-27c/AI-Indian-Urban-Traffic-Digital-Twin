import cv2


def draw_frame_info(
    frame,
    frame_number,
    fps,
):
    text = f"Frame: {frame_number} | FPS: {fps:.1f}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame