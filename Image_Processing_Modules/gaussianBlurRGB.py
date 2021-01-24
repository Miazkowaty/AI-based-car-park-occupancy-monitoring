import numpy


class gaussianBlurRGB:

    def __convolution(self, image, kernel):

        kernel_h = kernel.shape[0]
        kernel_w = kernel.shape[1]

        if len(image.shape) == 3:
            image_pad = numpy.pad(image, pad_width = ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2), (0, 0)), mode = 'constant', constant_values = 0).astype(numpy.float32)
        elif len(image.shape) == 2:
            image_pad = numpy.pad(image, pad_width = ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)), mode = 'constant', constant_values = 0).astype(numpy.float32)

        h = kernel_h // 2
        w = kernel_w // 2
        image_conv = numpy.zeros(image_pad.shape)

        # iteration through the image
        for i in range(h, image_pad.shape[0] - h):
            for j in range(w, image_pad.shape[1] - w):
                n = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
                n = n.flatten() * kernel.flatten()
                image_conv[i][j] = n.sum()

        h_end = - h
        w_end = - w

        if h == 0:
            return image_conv[h:, w:w_end]

        if w == 0:
            return image_conv[h:h_end, w:]

        image = image_conv[h:h_end, w:w_end]

        return image


    def __gaussian_blur(self, image, std_dev):

        image = numpy.asarray(image)
        filter_size = 2 * int(4 * std_dev + 0.5) + 1
        kernel = numpy.zeros((filter_size, filter_size), numpy.float32)
        m = filter_size // 2
        n = filter_size // 2

        # construction of the filter
        for x in range( - m, m + 1):
            for y in range( - n, n + 1):
                x1 = 2 * numpy.pi * (std_dev ** 2)
                x2 = numpy.exp( - (x ** 2 + y ** 2) / (2 * std_dev ** 2))
                kernel[x + m, y + n] = (1 / x1) * x2

        image = self.__convolution(image, kernel)

        return image


    def proceed(self, image, std_dev):

        # division into R, G and B factors
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        image[:, :, 0] = self.__gaussian_blur(r, std_dev)
        image[:, :, 1] = self.__gaussian_blur(g, std_dev)
        image[:, :, 2] = self.__gaussian_blur(b, std_dev)

        # conversion
        image = image.astype(numpy.uint8)

        return image