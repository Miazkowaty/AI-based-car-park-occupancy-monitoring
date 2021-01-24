class singleParkingSpace:

    def __init__(self, _ID, _image, _x, _y, _w, _h):

        self.ID = _ID
        self.image = _image[_y:_y + _h, _x:_x + _w]
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self.isOccupied = False

        return None