import os
import cv2
import math
from Classes.objectDetector import *
from Classes.singleParkingSpace import *
from Image_Processing_Modules.gaussianBlurRGB import *
from Image_Processing_Modules.histogramEqualizerRGB import *
from Image_Processing_Modules.nonlocalMeansDenoiserRGB import *


class carParkOccupancyMonitoring:

    def __init__(self, _inputVideoFile, _timeInterval, _color, _thickness, _use__gaussianBlurRGB, _use__histogramEqualizerRGB, _use__nonlocalMeansDenoiserRGB):

        self.inputVideoFile = _inputVideoFile
        self.timeInterval = _timeInterval
        r, g, b = _color[0], _color[1], _color[2]
        self.color = (b, g, r)
        self.thickness = _thickness
        self.use__gaussianBlurRGB = _use__gaussianBlurRGB
        self.use__histogramEqualizerRGB = _use__histogramEqualizerRGB
        self.use__nonlocalMeansDenoiserRGB = _use__nonlocalMeansDenoiserRGB
        self.gb_RGB = gaussianBlurRGB()
        self.he_RGB = histogramEqualizerRGB()
        self.md_RGB = nonlocalMeansDenoiserRGB()
        self.carDetector = objectDetector('Cascade_Model/haar_cascade_car_v5.xml', 1.0045, 500, (80, 80), (500, 500), 0.2, self.color, self.thickness)

        return None


    def do_the_thing(self):

        # removing output images from last use of the script
        for file in os.listdir('Output_Images'):

            os.remove(os.path.join('Output_Images', file))

        # uploading input file
        cap = cv2.VideoCapture('Input_Videos/' + self.inputVideoFile)

        # getting frame rate
        frameRate = cap.get(5)

        # main loop of the script
        while cap.isOpened():

            ret, currentFrame = cap.read()

            if ret == False:
                break

            # getting current frame number
            frameId = cap.get(1)

            if frameId % math.floor(self.timeInterval * frameRate) == 0:

                # saving original frame cut from source video
                filename1 = 'Output_Images/frame_' + str(int(frameId)) + '_original.jpg'
                cv2.imwrite(filename1, currentFrame)

                image = currentFrame

                # image processing
                if self.use__histogramEqualizerRGB:
                    image = self.he_RGB.proceed(image)

                if self.use__nonlocalMeansDenoiserRGB:
                    image = self.md_RGB.proceed(image)

                if self.use__gaussianBlurRGB:
                    image = self.gb_RGB.proceed(image, 1.1)

                # initialization of list of singleParkingSpace objects
                pSpace_y = int(image.shape[0] / 100 * 43)
                pSpace_w = int(image.shape[1] / 100 * 17)
                pSpace_h = int(image.shape[0] / 100 * 35)

                parking = [
                singleParkingSpace(1, image, int(image.shape[1] / 100 * 0), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(2, image, int(image.shape[1] / 100 * 12), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(3, image, int(image.shape[1] / 100 * 25), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(4, image, int(image.shape[1] / 100 * 39), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(5, image, int(image.shape[1] / 100 * 53), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(6, image, int(image.shape[1] / 100 * 67), pSpace_y, pSpace_w, pSpace_h),
                singleParkingSpace(7, image, int(image.shape[1] / 100 * 81), pSpace_y, pSpace_w, pSpace_h)]

                # iteration through all of registered parking spots
                for pSpace in parking:

                    # detection of cars in sub-images
                    pSpace.image, numOfDetections = self.carDetector.detect(pSpace.image)

                    if numOfDetections > 0:
                        pSpace.isOccupied = True
                    else:
                        pSpace.isOccupied = False

                # parking_info list components description:
                #   [0]: number of occupied parking spots
                #   [1]: ID's of occupied parking spots
                #   [2]: number of available parking spots
                #   [3]: ID's of available parking spots
                parking_info = [0, '', 0, '']

                # iteration through all of registered parking spots
                for pSpace in parking:

                    # gathering info-s about current state of parking
                    if pSpace.isOccupied == True:
                        parking_info[0] += 1
                        parking_info[1] += '[' + str(pSpace.ID) + '] '
                    else:
                        parking_info[2] += 1
                        parking_info[3] += '[' + str(pSpace.ID) + '] '

                    # placing info-s about current state of every parking spot in the base image
                    if pSpace.isOccupied == True:
                        image = cv2.circle(image, (pSpace.x + int(pSpace.w / 2), pSpace.y + pSpace.h + int(pSpace.h / 5)), int(pSpace.h / 25), (0, 0, 255), int(pSpace.h / 12))
                    else:
                        image = cv2.circle(image, (pSpace.x + int(pSpace.w / 2), pSpace.y + pSpace.h + int(pSpace.h / 5)), int(pSpace.h / 25), (0, 255, 0), int(pSpace.h / 12))

                # creating and placing text info-s in the base image and terminal, 1/2
                occupied_spots_info = 'occupied parking spots: ' + str(parking_info[0]) + '/' + str(len(parking)) + ';  ID\'s: ' + parking_info[1]
                image = cv2.putText(image, occupied_spots_info, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, self.thickness, cv2.LINE_AA) 
                print('\n' + occupied_spots_info)

                # creating and placing text info-s in the base image and terminal, 2/2
                available_spots_info = 'available parking spots: ' + str(parking_info[2]) + '/' + str(len(parking)) + ';  ID\'s: ' + parking_info[3]
                image = cv2.putText(image, available_spots_info, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, self.thickness, cv2.LINE_AA)
                print(available_spots_info)

                # saving processed copy of original frame cut from source video
                filename2 = 'Output_Images/frame_' + str(int(frameId)) + '.jpg'
                cv2.imwrite(filename2, image)

        cap.release()

        return None


app = carParkOccupancyMonitoring('car-parking-monitoring-1080p.mp4', 10, (255, 255, 255), 2, True, True, True)
app.do_the_thing()