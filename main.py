from imageclass import ImageClassifier
from imageclass import ImageClass

images_classifier = ImageClassifier(10, 2)

camel_images_train, camel_images_test = images_classifier.read_images('028.camel', 'camel')
dog_images_train, dog_images_test = images_classifier.read_images('056.dog', 'dog')
dolphin_images_train, dolphin_images_test = images_classifier.read_images('057.dolphin', 'dolphin')
giraffe_images_train, giraffe_images_test = images_classifier.read_images('084.giraffe', 'giraffe')
goose_images_train, goose_images_test = images_classifier.read_images('089.goose', 'goose')
horse_images_train, horse_images_test = images_classifier.read_images('105.horse', 'horse')

'''':arg image retrieval with rgb


images_classifier.create_rgb_histogram(camel_images_train)
images_classifier.create_rgb_histogram(dog_images_train)
images_classifier.create_rgb_histogram(dolphin_images_train)
images_classifier.create_rgb_histogram(giraffe_images_train)
images_classifier.create_rgb_histogram(goose_images_train)
images_classifier.create_rgb_histogram(horse_images_train)

images_classifier.create_rgb_histogram(camel_images_test)
images_classifier.create_rgb_histogram(dog_images_test)
images_classifier.create_rgb_histogram(dolphin_images_test)
images_classifier.create_rgb_histogram(giraffe_images_test)
images_classifier.create_rgb_histogram(goose_images_test)
images_classifier.create_rgb_histogram(horse_images_test)

train_images = camel_images_train + dog_images_train + dolphin_images_train + giraffe_images_train + goose_images_train + horse_images_train

for test_image in camel_images_test:
    images_classifier.show_images(test_image, images_classifier.calculate_five_most_similar_image(train_images, test_image))

for i, test_image in enumerate(dolphin_images_test):
    images_classifier.write_output_images(test_image, images_classifier.calculate_five_most_similar_image_with_rgb(train_images, test_image), 'dolphin', i)
'''



''':arg image retrieval with hsv histogram
'''
images_classifier.create_hsv_histogram(camel_images_train)
images_classifier.create_hsv_histogram(dog_images_train)
images_classifier.create_hsv_histogram(dolphin_images_train)
images_classifier.create_hsv_histogram(giraffe_images_train)
images_classifier.create_hsv_histogram(goose_images_train)
images_classifier.create_hsv_histogram(horse_images_train)

images_classifier.create_hsv_histogram(camel_images_test)
images_classifier.create_hsv_histogram(dog_images_test)
images_classifier.create_hsv_histogram(dolphin_images_test)
images_classifier.create_hsv_histogram(giraffe_images_test)
images_classifier.create_hsv_histogram(goose_images_test)
images_classifier.create_hsv_histogram(horse_images_test)

train_images = camel_images_train + dog_images_train + dolphin_images_train + giraffe_images_train + goose_images_train + horse_images_train


for i, test_image in enumerate(dolphin_images_test):
    images_classifier.write_output_images(test_image, images_classifier.calculate_five_most_similar_image_with_hsv(train_images, test_image), 'dolphin', i)