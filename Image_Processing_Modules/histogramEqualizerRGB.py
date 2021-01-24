import numpy


class histogramEqualizerRGB:

    def __enhance_contrast(self, image):

        bin_num = 256
        image_flattened = image.flatten()
        image_histogram = numpy.zeros(bin_num)

        # calculating the frequency
        for pixel in image:
            image_histogram[pixel] += 1

        # cummulative sum
        cum_sum = numpy.cumsum(image_histogram)
        norm = (cum_sum - cum_sum.min()) * (bin_num - 1)

        # normalization of the pixel values
        normalized = norm / (cum_sum.max() - cum_sum.min())
        normalized = normalized.astype('int')

        # flat histogram
        image_eq = normalized[image_flattened]

        # reshaping the flattened matrix
        image = numpy.reshape(a = image_eq, newshape = image.shape)

        return image


    def proceed(self, image):

        # division into R, G and B factors
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        r_eq = self.__enhance_contrast(r)
        g_eq = self.__enhance_contrast(g)
        b_eq = self.__enhance_contrast(b)

        # combination of factors
        image = numpy.dstack((r_eq, g_eq, b_eq))

        # conversion
        image = image.astype(numpy.uint8)

        return image