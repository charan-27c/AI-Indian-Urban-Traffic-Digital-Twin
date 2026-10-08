from collections import defaultdict



class VehicleHistory:


    def __init__(self):

        self.history = defaultdict(list)



    def update(
        self,
        vehicles
    ):


        for vehicle in vehicles:


            vehicle_id = (
                vehicle["vehicle_id"]
            )


            x1,y1,x2,y2 = (
                vehicle["bbox"]
            )


            center = (
                (x1+x2)//2,
                (y1+y2)//2
            )


            self.history[
                vehicle_id
            ].append(
                center
            )



    def get_history(
        self,
        vehicle_id
    ):

        return self.history[
            vehicle_id
        ]