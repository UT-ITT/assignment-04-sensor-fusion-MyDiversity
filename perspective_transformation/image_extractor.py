import cv2
from pathlib import Path

# Path Setup
THIS_DIR = Path(__file__).resolve().parent
PREVIEW_WINDOW = 'Preview Window'
CROPPED_WINDOW = 'Cropped Image'

print('Please enter the path to the image you want to open. i.e. user/images/sample_image.jpg')
image_path = input(': ').strip()

# load image and exit if an error occures
img = cv2.imread(str(image_path))

if img is None:
    print('Error: The image could not be loaded')
    exit()

# current cutout
cropped_img = None

# working copy of original image
img_copy = img.copy()

# List to store selected points
corner_coordinates = []

# Function to reset cutout selection
def reset_selection():
    global img_copy, corner_coordinates, cropped_img

    img_copy = img.copy()
    corner_coordinates.clear()
    cropped_img = None

    cv2.imshow(PREVIEW_WINDOW, img_copy)

def mouse_callback(event, x, y, flags, param):
    global img_copy, corner_coordinates, cropped_img

    if event == cv2.EVENT_LBUTTONDOWN:
        # only register at most 4 clicks
        if len(corner_coordinates) < 4:
            corner_coordinates.append((x, y))

            # draw corner points
            cv2.circle(img_copy, (x, y), 5, (50, 50, 255), -1)

            # connect corners for better visualization
            if len(corner_coordinates) > 1:
                cv2.line(img_copy, corner_coordinates[-2], corner_coordinates[-1], (100, 100, 255), 2)

            # Crop the image after 4 points were selected
            if len(corner_coordinates) == 4:
                cv2.line(img_copy, corner_coordinates[3], corner_coordinates[0], (100, 100, 255), 2)
            
                # split x and y coordinates
                x_coords = [p[0] for p in corner_coordinates]
                y_coords = [p[1] for p in corner_coordinates]

                # get bounding box of the selected points 
                x_min = max(0, min(x_coords))
                x_max = min(img.shape[1], max(x_coords))

                y_min = max(0, min(y_coords)) 
                y_max = min(img.shape[0], max(y_coords))

                # crop image
                cropped_img = img[y_min:y_max, x_min:x_max]

                # show cropped image
                cv2.imshow(CROPPED_WINDOW, cropped_img)

                print("Image cropped! 's' = save, 'ESC' = discard, 'q' = exit")
            
            cv2.imshow(PREVIEW_WINDOW, img_copy)
                    
# Create window
cv2.namedWindow(PREVIEW_WINDOW)
cv2.imshow(PREVIEW_WINDOW, img_copy)
cv2.setMouseCallback(PREVIEW_WINDOW, mouse_callback)

while True:
    key = cv2.waitKey(20) & 0xFF

    # Exit script
    if key == ord('q'):
        break               

    # If cutout has been selected
    if cropped_img is not None:

        # save 
        if key == ord('s'):
            print('Select a path to store the image. i.e. user/images/cropped_image.jpg')
            save_path = input(': ').strip()
            print('Please enter your preferred image size')
            image_width = input('Width: ')
            image_height = input('Height: ')
            
            resized_img = cv2.resize(cropped_img, (int(image_width), int(image_height)), interpolation = cv2.INTER_CUBIC)

            cv2.imwrite(str(save_path), resized_img)
            print(f'Image saved at {save_path}')

            cv2.destroyWindow(CROPPED_WINDOW)
            reset_selection()
        
        # discard
        elif key == 27:  #ESC
            print("selection discarded")

            cv2.destroyWindow(CROPPED_WINDOW)
            reset_selection()

cv2.destroyAllWindows()
