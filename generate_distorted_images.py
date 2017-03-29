import cv2
import os
import sys
from DisplayUtils.TileDisplay import show_img
from ImageProcessing.OpenCVUtils import rotate_image


def dilate_img(img, file_name, file_folder, show, write):
    for iterations in range(1, 4, 2):
        for dilate in range(1, 4, 2):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
            dilated = cv2.dilate(img, kernel, iterations=iterations)
            for rot in range(-2, 3, 2):
                dst = rotate_image(dilated, rot)
                title = 'dilated-r-' + str(rot) + '-d-' + str(dilate) + '-i-' + str(iterations)
                if show:
                    show_img(title, dst)
                if write:
                    print 'Writing ' + file_folder + '/' + file_name + title + '.png'
                    cv2.imwrite(file_folder + '/' + file_name + title + '.png', dst)


def erode_img(img, file_name, file_folder, show, write):
    for iterations in range(1, 3):
        for erode in range(1, 4, 2):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
            eroded = cv2.erode(img, kernel, iterations=iterations)
            for rot in range(-2, 3, 2):
                dst = rotate_image(eroded, rot)
                title = 'eroded-r-' + str(rot) + '-e-' + str(erode) + '-i-' + str(iterations)
                if show:
                    show_img(title, dst)
                if write:
                    cv2.imwrite(file_folder + '/' + file_name + title + '.png', dst)


def process_image(path, show=True, write=False):
    print path
    img = cv2.imread(path)
    if show:
        show_img('orig', img)

    file_folder, full_file = os.path.split(path)
    file_name = full_file.split('.')[0]

    # erode_img(img, file_name, file_folder, show, write)
    dilate_img(img, file_name, file_folder, show, write)


def show_distorted(path, show, write):
    process_image(path, show=show, write=write)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_directory(folder):
    for file_name in os.listdir(folder):
        if not file_name.startswith('.'):
            process_image(folder + file_name, show=False, write=True)


def main():
    img_file = 'training/7/7_1079_crop_3.png'
    show = True

    if len(sys.argv) == 2:
        img_file = sys.argv[1]
        show = False

    show_distorted(img_file, show, True)
    # For use if you want to generate extra images for all files in a directory
    # process_directory('training/8/')


if __name__ == "__main__":
    main()
