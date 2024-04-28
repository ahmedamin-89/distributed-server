import cv2

def process_image(operation, filename):
    """ Process the image based on the operation specified. """
    image = cv2.imread(filename)
    if image is None:
        return None, "Error: Image not found."
    if operation == "grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif operation == "blur":
        result = cv2.GaussianBlur(image, (5, 5), 0)
    elif operation == "edges":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
    else:
        return None, "Error: Unknown operation."
    output_filename = f"processed_{filename}"
    cv2.imwrite(output_filename, result)
    return output_filename, "Success"
