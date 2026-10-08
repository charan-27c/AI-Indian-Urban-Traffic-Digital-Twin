def get_center(bbox):

    x1, y1, x2, y2 = bbox

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )