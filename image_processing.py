import cv2
import numpy as np

def process_image(operation, filename):
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

def advanced_process_image(comm, operation, filename):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        image = cv2.imread(filename)
        height, width, _ = image.shape
        chunk_height = height // size
        chunks = [image[i * chunk_height:(i + 1) * chunk_height, :] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    if operation == "segmentation":
        pixel_values = chunk.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_chunk = centers[labels.flatten()]
        segmented_chunk = segmented_chunk.reshape(chunk.shape)
    else:
        segmented_chunk = chunk  # Placeholder for other operations

    gathered_chunks = comm.gather(segmented_chunk, root=0)

    if rank == 0:
        result = np.vstack(gathered_chunks)
        output_filename = f"processed_{filename}"
        cv2.imwrite(output_filename, result)
        return output_filename, "Success"
    else:
        return None, "Processing"

