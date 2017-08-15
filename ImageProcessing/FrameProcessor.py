import cv2
import numpy as np
import os
from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
CROP_DIR = 'crops'


class FrameProcessor:
    def __init__(self, height, version, debug=False, write_digits=False):
        self.debug = debug
        self.version = version
        self.height = height
        self.file_name = None
        self.img = None
        self.width = 0
        self.original = None
        self.write_digits = write_digits

        self.knn = self.train_knn(self.version)

    def set_image(self, file_name):
        self.file_name = file_name
        self.img = cv2.imread(file_name)
        self.original, self.width = self.resize_to_height(self.height)
        self.img = self.original.copy()

    def resize_to_height(self, height):
        r = self.img.shape[0] / float(height)
        dim = (int(self.img.shape[1] / r), height)
        img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        return img, dim[0]

    def train_knn(self, version):
        npa_classifications = np.loadtxt("knn/classifications" + version + ".txt",
                                         np.float32)  # read in training classifications
        npa_flattened_images = np.loadtxt("knn/flattened_images" + version + ".txt",
                                          np.float32)  # read in training images

        npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)
        return k_nearest

    def process_image(self, blur, threshold, adjustment, erode, iterations):

        self.img = self.original.copy()

        debug_images = []

        alpha = float(2.5)

        debug_images.append(('Original', self.original))

        # Adjust the exposure
        exposure_img = cv2.multiply(self.img, np.array([alpha]))
        debug_images.append(('Exposure Adjust', exposure_img))

        # Convert to grayscale
        img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)
        debug_images.append(('Grayscale', img2gray))

        # Blur to reduce noise
        img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)
        debug_images.append(('Blurred', img_blurred))

        cropped = img_blurred

        # Threshold the image
        cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  threshold, adjustment)
        debug_images.append(('Cropped Threshold', cropped_threshold))

        # Erode the lcd digits to make them continuous for easier contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', eroded))

        # Reverse the image to so the white text is found when looking for the contours
        inverse = inverse_colors(eroded)
        debug_images.append(('Inversed', inverse))

        # Find the lcd digit contours
        _, contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

        # Assuming we find some, we'll sort them in order left -> right
        if len(contours) > 0:
            contours, _ = sort_contours(contours)

        potential_decimals = []
        potential_digits = []

        total_digit_height = 0
        total_digit_y = 0

        # Aspect ratio for all non 1 character digits
        desired_aspect = 0.6
        # Aspect ratio for the "1" digit
        digit_one_aspect = 0.3
        # The allowed buffer in the aspect when determining digits
        aspect_buffer = 0.15

        # Loop over all the contours collecting potential digits and decimals
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h

            # It's a square, save the contour as a potential digit
            if size > 100 and aspect >= 1 - .3 and aspect <= 1 + .3:
                potential_decimals.append(contour)

            # If it's small and it's not a square, kick it out
            if size < 20 * 100 and (aspect < 1 - aspect_buffer and aspect > 1 + aspect_buffer):
                continue

            # Ignore any rectangles where the width is greater than the height
            if w > h:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                continue

            # If the contour is of decent size and fits the aspect ratios we want, we'll save it
            if ((size > 2000 and aspect >= desired_aspect - aspect_buffer and aspect <= desired_aspect + aspect_buffer) or
                (size > 1000 and aspect >= digit_one_aspect - aspect_buffer and aspect <= digit_one_aspect + aspect_buffer)):
                # Keep track of the height and y position so we can run averages later
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)
        left_most_digit = 0
        right_most_digit = 0
        digit_x_positions = []

        # Calculate the average digit height and y position so we can determine what we can throw out
        if potential_digits_count > 0:
            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            if self.debug:
                print("Average Digit Height and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))

        output = ''
        ix = 0

        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        for pot_digit in potential_digits:
            [x, y, w, h] = cv2.boundingRect(pot_digit)

            # Does this contour match the averages
            if h <= avg_digit_height * 1.2 and h >= avg_digit_height * 0.2 and y <= avg_digit_height * 1.2 and y >= avg_digit_y * 0.2:
                # Crop the contour off the eroded image
                cropped = eroded[y:y + h, x: x + w]
                # Draw a rect around it
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                debug_images.append(('digit' + str(ix), cropped))

                # Call into the KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit: " + digit)
                output += digit

                # Helper code to write out the digit image file for use in KNN training
                if self.write_digits:
                    _, full_file = os.path.split(self.file_name)
                    file_name = full_file.split('.')
                    crop_file_path = CROP_DIR + '/' + digit + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    cv2.imwrite(crop_file_path, cropped)

                # Track the x positions of where the digits are
                if left_most_digit == 0 or x < left_most_digit:
                    left_most_digit = x

                if right_most_digit == 0 or x > right_most_digit:
                    right_most_digit = x + w

                digit_x_positions.append(x)

                ix += 1
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (66, 146, 244), 2)

        decimal_x = 0
        # Loop over the potential digits and find a square that's between the left/right digit x positions on the
        # lower half of the screen
        for pot_decimal in potential_decimals:
            [x, y, w, h] = cv2.boundingRect(pot_decimal)

            if x < right_most_digit and x > left_most_digit and y > (self.height / 2):
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                decimal_x = x

        # Once we know the position of the decimal, we'll insert it into our string
        for ix, digit_x in enumerate(digit_x_positions):
            if digit_x > decimal_x:
                # insert
                output = output[:ix] + '.' + output[ix:]
                break

        # Debugging to show the left/right digit x positions
        if self.debug:
            cv2.rectangle(self.img, (left_most_digit, int(avg_digit_y)),
                          (left_most_digit + right_most_digit - left_most_digit,
                           int(avg_digit_y) + int(avg_digit_height)),
                          (66, 244, 212), 2)

        # Log some information
        if self.debug:
            print("Potential Digits " + str(len(potential_digits)))
            print("Potential Decimals " + str(len(potential_decimals)))
            print("String: " + output)

        return debug_images, output

    # Predict the digit from an image using KNN
    def predict_digit(self, digit_mat):
        # Resize the image
        imgROIResized = cv2.resize(digit_mat, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Reshape the image
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert it to floats
        npaROIResized = np.float32(npaROIResized)
        _, results, neigh_resp, dists = self.knn.findNearest(npaROIResized, k=1)
        predicted_digit = str(chr(int(results[0][0])))
        if predicted_digit == 'A':
            predicted_digit = '.'
        return predicted_digit
