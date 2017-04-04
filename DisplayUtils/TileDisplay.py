import cv2

start_y = 20
current_stack_y = start_y
current_stack_x = 0
current_stack_width = 0
min_width = 200
max_stack_y = 450
y_buffer = 25
x_buffer = 10


def show_img(name, img):
    global current_stack_y, current_stack_x, current_stack_width
    height, width = img.shape[:2]
    if width < min_width:
        width = min_width
    if width > current_stack_width:
        current_stack_width = width
    cv2.imshow(name, img)
    cv2.moveWindow(name, current_stack_x, current_stack_y)
    current_stack_y += height + y_buffer
    if current_stack_y > max_stack_y:
        current_stack_y = start_y
        current_stack_x += current_stack_width + x_buffer


def reset_tiles():
    global current_stack_x, current_stack_y, current_stack_width
    current_stack_x = 0
    current_stack_y = current_stack_y
    current_stack_width = 0
