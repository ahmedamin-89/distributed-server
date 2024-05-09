import cv2
import numpy as np
import io
from flask import Flask, request, send_file
from flask_cors import CORS
from mpi4py import MPI
import math
import time

app = Flask(__name__)
CORS(app)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_image(image, num_parts):
    height, width = image.shape[:2]
    part_height = height // num_parts
    part_width = width // num_parts
    parts = []

    for i in range(num_parts):
        for j in range(num_parts):
            part = image[i * part_height: (i + 1) * part_height, j * part_width: (j + 1) * part_width]
            parts.append(part)

    return parts

def combine_image(parts, num_parts, original_shape):
    combined_image = np.zeros(original_shape, dtype=np.uint8)
    part_height = original_shape[0] // num_parts
    part_width = original_shape[1] // num_parts

    for i in range(num_parts):
        for j in range(num_parts):
            combined_image[i * part_height: (i + 1) * part_height, j * part_width: (j + 1) * part_width] = parts[i * num_parts + j]

    return combined_image

@app.route('/process_image', methods=['POST'])
def process_image():
    if rank == 0:  # Master node
        print("Master node receiving image and operation...")
        # Receive image and operation from client
        image_data = request.files['image'].read()
        operation = request.form['operation']

        # Convert binary image data to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        """
        # Decode NumPy array to image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Slice image into (size) parts
        image_parts = np.array_split(image, size, axis=0)

        # Return back to original binary format, as received.
        for idx, slice in enumerate(image_parts): 
            retval, buffer = cv2.imencode('.jpg', slice)
            image_parts[idx] = buffer.tobytes()

        print("Splitting image into parts...")
        # Split the image into equal parts based on the number of worker nodes
        # chunk_size = math.ceil(len(image_data) / size)
        # image_parts = [image_data[i:i+chunk_size] for i in range(0, len(image_data), chunk_size)]
        """

        image_parts = split_image(nparr, size)

        print("Distributing tasks to worker nodes...")
        # Distribute tasks to worker nodes
        for i in range(1, size):
            print("Sending task to worker node " + str(i))
            comm.send((operation, image_parts[i]), dest=i)

        print("Processing master's own task...")
        # Process the master's own task
        processed_image_data = process_image_task(operation, image_parts[0])

        print("Collecting processed images from worker nodes...")
        # Collect processed images from worker nodes
        processed_image_parts = []
        #for i in range(1, size):
        #    processed_image_data += comm.recv(source=i)

        for i in range(1, size):
            processed_image_part = comm.recv(source=i)
            processed_image_parts.append(processed_image_part)

        # Concatenate processed image parts along the vertical axis
        """
        processed_image_data = np.concatenate(processed_image_parts, axis=0)
        """

        processed_image_data = combine_image(processed_image_parts, size, image.shape)

        print("Sending processed image data back to client...")
        _, processed_image_data = cv2.imencode('.jpg', processed_image_data)
        # Return processed image data
        return send_file(
            io.BytesIO(processed_image_data),
            mimetype='image/jpeg'
        )
    elif 1 == 2:  # Worker nodes
        time.sleep(10)
        # Receive task from master node
        operation, image_data = comm.recv(source=0)

        print(f"Worker node {rank} receiving task...")
        # Process image task
        processed_image_data = process_image_task(operation, image_data)

        print(f"Worker node {rank} sending processed image data back to master...")
        # Send processed image back to master node
        comm.send(processed_image_data, dest=0)

def process_image_task(operation, image_data):
    # Process image using OpenCV
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Perform image processing operation based on 'operation'
    if operation == 'grayscale':
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif operation == 'color_inversion':
        processed_image = cv2.bitwise_not(image)
    else:
        processed_image = image

    # Encode processed image to JPEG format
    #_, processed_image_data = cv2.imencode('.jpg', processed_image)

    # Return processed image data
    #return processed_image_data
    return processed_image

if __name__ == '__main__':
    if rank == 0:
        print("Master node starting Flask application...")
        app.run(host='0.0.0.0', port=5000)
    else:  # Worker nodes
        while True:
            image_data = None
            while True:
            # Receive task from master node
                operation, image_data = comm.recv(source=0)

                # Check if the received task is a termination signal
                if image_data is not None:
                    break  # Break the loop if termination signal is received

            print(f"Worker node {rank} receiving task...")
            # Process image task
            processed_image_data = process_image_task(operation, image_data)

            # Send processed image back to master node
            comm.send(processed_image_data, dest=0)

            # Print termination message after the loop
            print(f"Worker node {rank} has terminated.")