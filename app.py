import os
import cv2
import numpy as np
import io
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from mpi4py import MPI
import time

app = Flask(__name__, static_folder='./dist', static_url_path='/')
CORS(app)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_image(image, num_parts):
    height, width = image.shape[:2]
    part_height = height // num_parts
    parts = [image[i * part_height: (i + 1) * part_height] for i in range(num_parts)]
    return parts

def combine_image(parts, original_shape):
    return np.vstack(parts)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if rank == 0:  # Master node
        print("[Master] Receiving image and operation from client...")
        # Receive image and operation from client
        image_file = request.files['image']
        operation = request.form['operation']

        # Check file size (10 MB = 10 * 1024 * 1024 bytes)
        if len(image_file.read()) > 10 * 1024 * 1024:
            print("[Master] Error: File size exceeds 10 MB")
            return jsonify({'error': 'File size exceeds 10 MB'}), 400

        # Reset file stream position to start
        image_file.seek(0)

        image_data = image_file.read()

        # Convert binary image data to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Split the image into equal parts based on the number of worker nodes
        image_parts = split_image(image, size)
        print(f"[Master] Image split into {size} parts.")

        print("[Master] Distributing tasks to worker nodes...")
        # Distribute tasks to worker nodes
        for i in range(1, size):
            print(f"[Master] Sending task to worker node {i}")
            comm.send((operation, image_parts[i]), dest=i)

        print("[Master] Processing its own part...")
        # Process the master's own task
        processed_image_part = process_image_task(operation, image_parts[0])

        print("[Master] Collecting processed images from worker nodes...")
        # Collect processed images from worker nodes
        processed_image_parts = [processed_image_part]
        for i in range(1, size):
            print(f"[Master] Receiving processed part from worker node {i}")
            processed_image_parts.append(comm.recv(source=i))

        # Combine processed image parts
        processed_image = combine_image(processed_image_parts, image.shape)
        print("[Master] All parts received and combined.")

        print("[Master] Sending processed image data back to client...")
        _, processed_image_data = cv2.imencode('.jpg', processed_image)
        # Return processed image data
        return send_file(
            io.BytesIO(processed_image_data),
            mimetype='image/jpeg'
        )
    else:  # Worker nodes
        while True:
            print(f"[Worker {rank}] Waiting for tasks...")
            # Receive task from master node
            operation, image_part = comm.recv(source=0)
            print(f"[Worker {rank}] Task received. Processing...")

            # Process image task
            processed_image_part = process_image_task(operation, image_part)

            print(f"[Worker {rank}] Sending processed image part back to master...")
            # Send processed image back to master node
            comm.send(processed_image_part, dest=0)

def process_image_task(operation, image):
    print(f"[Process] Performing operation: {operation}")
    # Perform image processing operation based on 'operation'
    if operation == 'grayscale':
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    elif operation == 'blur':
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif operation == 'color_inversion':
        processed_image = cv2.bitwise_not(image)
    elif operation == 'threshold_segmentation':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    elif operation == 'adaptive_threshold_segmentation':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    else:
        processed_image = image

    print(f"[Process] Operation {operation} completed.")
    return processed_image

if __name__ == '__main__':
    if rank == 0:
        print("[Master] Starting Flask application...")
        app.run(host='0.0.0.0', port=5000)
    else:  # Worker nodes
        while True:
            # Worker node waits for tasks
            image_data = None
            while True:
                # Receive task from master node
                operation, image_data = comm.recv(source=0)
                if image_data is not None:
                    break  # Break the loop if task received

            print(f"[Worker {rank}] Processing task...")
            # Process image task
            processed_image_part = process_image_task(operation, image_data)

            print(f"[Worker {rank}] Sending processed image data back to master...")
            # Send processed image back to master node
            comm.send(processed_image_part, dest=0)
