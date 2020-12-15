import glob, os
from PIL import Image
import numpy as np
import random

PIXEL_CONST = 400

class ImageClass:
    def __init__(self, image):
        self.image = image
        self.width, self.height = image.size
        self.r_histogram = np.zeros(256, dtype=np.float64)
        self.g_histogram = np.zeros(256, dtype=np.float64)
        self.b_histogram = np.zeros(256, dtype=np.float64)
        self.h_histogram = np.zeros(256, dtype=np.float64)
        self.s_histogram = np.zeros(256, dtype=np.float64)
        self.v_histogram = np.zeros(256, dtype=np.float64)


class ImageClassifier:
    def __init__(self, train_image_count, test_image_count):
        self.test_image_count = test_image_count
        self.train_image_count = train_image_count

    def read_images(self, file_path, tag):
        images = []
        train_images = []
        test_images = []
        random_indexes = []
        for file in glob.glob('images\\' + file_path + '\\*.jpg'):
            im = (Image.open(file))
            images.append(ImageClass(im))

        test_count = 0
        while test_count < self.test_image_count:
            index = random.randint(0, len(images) -1 )
            if (index not in random_indexes) and images[index].width < PIXEL_CONST and images[index].height < PIXEL_CONST:
                random_indexes.append(index)
                test_images.append(images[index])
                images[index].image.save('inputs\\test_images_' + tag + '_' + str(test_count) + '.jpg')
                test_count += 1

        train_count = 0
        while train_count < self.train_image_count:
            index = random.randint(0, len(images) -1 )
            if (index not in random_indexes) and images[index].width < PIXEL_CONST and images[index].height < PIXEL_CONST:
                random_indexes.append(index)
                train_images.append(images[index])
                images[index].image.save('inputs\\train_images_' + tag + '_' + str(train_count) + '.jpg')
                train_count += 1

        return train_images, test_images


    def create_hsv_histogram(self, images):
        for image in images:
            print(' height : ' + str(image.height) + ' width : ' + str(image.width))
            for i in range(0, image.width):
                for j in range(0, image.height):
                    h, s, v = image.image.convert('HSV').getpixel((i, j))
                    image.h_histogram[h] += (1.0 / (image.width * image.height))
                    image.s_histogram[s] += (1.0 / (image.width * image.height))
                    image.v_histogram[v] += (1.0 / (image.width * image.height))


    def create_rgb_histogram(self, images):
        for image in images:
            print(' height : ' + str(image.height) + ' width : ' + str(image.width))
            for i in range(0, image.width):
                for j in range(0, image.height):
                    r, g, b = image.image.convert('RGB').getpixel((i, j))
                    image.r_histogram[r] += (1.0 / (image.width * image.height))
                    image.g_histogram[g] += (1.0 / (image.width * image.height))
                    image.b_histogram[b] += (1.0 / (image.width * image.height))
        print('over')


    def calculate_euclidean_distance(self, histogram_x, histogram_y):
        distance = [np.power(a_i - b_i, 2) for a_i, b_i in zip(histogram_x, histogram_y)]
        return np.sqrt(np.sum(distance))


    def calculate_five_most_similar_image_with_rgb(self, train_images, test_image):
        distance_matrix = np.zeros((len(train_images), 2), dtype=object)
        for i, train in enumerate(train_images):
            distance_r = self.calculate_euclidean_distance(test_image.r_histogram, train.r_histogram)
            distance_g = self.calculate_euclidean_distance(test_image.g_histogram, train.g_histogram)
            distance_b = self.calculate_euclidean_distance(test_image.b_histogram, train.b_histogram)
            distance = np.sqrt(distance_r ** 2 + distance_g ** 2 + distance_b ** 2)
            distance_matrix[i][0] = train
            distance_matrix[i][1] = np.float64(distance)
        distance_matrix = distance_matrix[distance_matrix[:, 1].argsort()]
        return distance_matrix[0: 5, :]


    def calculate_five_most_similar_image_with_hsv(self, train_images, test_image):
        distance_matrix = np.zeros((len(train_images), 2), dtype=object)
        for i, train in enumerate(train_images):
            distance_r = self.calculate_euclidean_distance(test_image.h_histogram, train.r_histogram)
            distance_g = self.calculate_euclidean_distance(test_image.s_histogram, train.g_histogram)
            distance_b = self.calculate_euclidean_distance(test_image.v_histogram, train.b_histogram)
            distance = np.sqrt(distance_r ** 2 + distance_g ** 2 + distance_b ** 2)
            distance_matrix[i][0] = train
            distance_matrix[i][1] = np.float64(distance)
        distance_matrix = distance_matrix[distance_matrix[:, 1].argsort()]
        return distance_matrix[0: 5, :]


    def write_output_images(self, file_path, test_image, distance_matrix, tag, index):
        test_image.image.show()
        test_image.image.save(file_path + '\\' + tag + '_' + str(index) + '_test_image.jpg')
        for i, row in enumerate(distance_matrix):
            row[0].image.show()
            print('distance :' + str(row[1]))
            row[0].image.save(file_path + '\\' + tag + '_' + str(index) + 'most_similar_' + str(i) + '.jpg')


