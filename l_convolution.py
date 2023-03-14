import numpy as np

class Convolution:
    def __init__(self, num_filters, nominal_filters):
        self.num_filters = num_filters

        if isinstance(nominal_filters, str) == True:
            self.filters = np.random.randn(num_filters, 3, 3) / 9
            
        else:
            self.filters = nominal_filters

    def iterate_image(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                image_region = image[i : (i + 3), j : (j + 3)]

                yield image_region, i, j

    def forward_propagation(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for image_region, i, j, in self.iterate_image(input):
            output[i, j] = np.sum(image_region * self.filters, axis = (1, 2))

        return output

    def back_propagation(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for image_region, i, j in self.iterate_image(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * image_region

        self.filters -= learn_rate * d_L_d_filters

        return None
