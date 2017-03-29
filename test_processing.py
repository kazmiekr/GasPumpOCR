import os
import time

from DisplayUtils.Colors import bcolors
from ImageProcessing import FrameProcessor, ProcessingVariables

std_height = 90

# thresh = 73  # 1-50 mod 2 25
# erode = 3  # 3-4 2
# adjust = 15  # 10-40 30
# blur = 9  # 5-15 mod 2 7

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur

version = '_2_0'
test_folder = 'tests/single_line'

frameProcessor = FrameProcessor(std_height, version, False, write_digits=False)


def test_img(path, expected, show_result=True):
    frameProcessor.set_image(path)
    (debug_images, calculated) = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)

    if expected == calculated:
        if show_result:
            print(bcolors.OKBLUE + 'Testing: ' + path + ', Expected ' + expected + ' and got ' + calculated + bcolors.ENDC)
        return True
    else:
        if show_result:
            print(bcolors.FAIL + 'Testing: ' + path + ', Expected ' + expected + ' and got ' + calculated + bcolors.ENDC)
        return False


def get_expected_from_filename(filename):
    expected = filename.split('.')[0]
    expected = expected.replace('A', '.')
    expected = expected.replace('Z', '')
    return expected


def run_tests(show_result=True):
    count = 0
    correct = 0

    start_time = time.time()
    for file_name in os.listdir(test_folder):
        # Skip hidden files
        if not file_name.startswith('.'):
            count += 1
            expected = get_expected_from_filename(file_name)
            is_correct = test_img(test_folder + '/' + file_name, expected, show_result)
            if is_correct:
                correct += 1

    print("\nFiles tested: " + str(count))
    print("Files correct: " + str(correct))
    acc = round(float(correct) / count * 100, 2)
    print("Test Params - erode: " + str(erode) + ", blur: " + str(blur) + ", adjust: " + str(
        adjustment) + ", thres: " + str(threshold) + ", iterations: " + str(iterations))
    print("Test Accuracy: " + bcolors.BOLD + str(acc) + '%' + bcolors.ENDC)
    return acc


def bulk_run():
    start_time = time.time()

    max_acc = 0
    best_erode = 0

    global erode, blur, adjustment, threshold, iterations
    for test_erode in range(3, 5):
        erode = test_erode
        for test_blur in range(5, 11, 2):
            blur = test_blur
            for test_adjust in range(10, 40, 2):
                adjustment = test_adjust
                for test_thres in range(11, 201, 2):
                    threshold = test_thres
                    for test_iterations in range(2, 5, 1):
                        iterations = test_iterations

                        acc = run_tests(show_result=False)
                        if acc > max_acc:
                            best_thres = threshold
                            best_erode = erode
                            best_blur = blur
                            best_adjust = adjustment
                            max_acc = acc
                            best_iterations = iterations

                        print("Best Params - erode: " + str(best_erode) + ", blur: " + str(best_blur) + ", adjust: " + str(
                            best_adjust) + ", thres: " + str(best_thres) + ", iterations: " + str(best_iterations))
                        print("Best Accuracy: " + bcolors.BOLD + str(max_acc)) + bcolors.ENDC

    print("--- %s seconds ---" % (time.time() - start_time))


def main():
    start_time = time.time()
    acc = run_tests()
    print("--- %s seconds ---" % (time.time() - start_time))

    # bulk_run()


if __name__ == "__main__":
    main()
