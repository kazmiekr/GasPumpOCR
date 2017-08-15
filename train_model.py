# train_model.py

import os
import cv2
import numpy as np

# Current version of training
version = '_2_1'

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

int_classifications = []
npa_flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
npa_classifications = []

trained_folder = 'knn'
trained_json_path = 'training' + version + '.json'


# Classify a digit
def train_file(file_path, char):
    global npa_flattened_images, int_classifications, npaRawFlattenedImages
    if char == 'dot':
        char = 'A'
    img = cv2.imread(file_path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThreshCopy = imgGray.copy()
    imgROIResized = cv2.resize(imgThreshCopy, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

    int_classifications.append(ord(char))
    npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    npa_flattened_images = np.append(npa_flattened_images, npaFlattenedImage, 0)


# Write out the dictionary as a string
def serialize_dict(dict):
    output = '{'
    count = 1
    proplen = len(dict)
    for key in dict:
        vals = dict[key]
        output += '"{}": {}'.format(key, vals)
        if count < proplen:
            output += ','
        count += 1
    output += '}'
    return output


# Write out the image mat data, to the format OpenCV expects
def serialize_mat(mat):
    type_id = 'opencv-matrix'
    dt = 'f'  # TODO: Be smarter about the data type
    return '{{"type_id":"{}", "dt":"{}", "data":{}, "rows":{}, "cols": {}}}\n' \
        .format(type_id, dt, serialize_array(mat), mat.shape[0], mat.shape[1])


# Write out an array into a string for use in serialization
def serialize_array(arr):
    output = '['
    for value in arr:
        for element in value:
            output += str(element) + ',' #'%.18e' % thing2 + ',\n'
    output = output[:-1]
    output += ']'
    return output


def main():
    training_dir = "training"

    for fname in os.listdir(training_dir):
        path = os.path.join(training_dir, fname)
        if os.path.isdir(path):
            print('Training ' + fname)
            tfiles = os.listdir(path)
            for tfile in tfiles:
                if not tfile.startswith('.'):
                    train_file(path + '/' + tfile, fname)

    # Save the classifications for use in Python
    fltClassifications = np.array(int_classifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
    np.savetxt(trained_folder + "/classifications" + version + ".txt", npaClassifications)
    np.savetxt(trained_folder + "/flattened_images" + version + ".txt", npa_flattened_images)

    # Save the classifications into a JSON file for use in C++/iOS
    data = {
        'classifications': serialize_mat(npaClassifications),
        'flattened_images': serialize_mat(npa_flattened_images)
    }
    with open(trained_folder + '/' + trained_json_path, 'w') as outfile:
        outfile.write(serialize_dict(data))


if __name__ == "__main__":
    main()
