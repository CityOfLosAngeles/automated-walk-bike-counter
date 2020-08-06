class LOIInfo:
    def __init__(self):
        self.mask_image = None
        # Points that are used in LOI GUI to create preview of the line for user
        self.preview_points = []
        # Points that are used to show the line base on image with video dimensions
        self.points = []
        self.side_A_name = ""
        self.side_B_name = ""
        self.sides_have_name = False
