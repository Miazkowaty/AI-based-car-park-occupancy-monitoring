import cv2


class nonlocalMeansDenoiserRGB:

    def proceed(self, image):

        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        return image