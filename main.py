from flask import Flask, request, send_file, jsonify
from mpi4py import MPI
import os
from image_processing import process_image, advanced_process_image

app = Flask(__name__)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if rank != 0:
        return jsonify({"error": "This endpoint can only be accessed by the master node."}), 403
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    operation = request.form.get('operation')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    if operation in ['grayscale', 'blur', 'edges']:
        processed_filename, status = process_image(operation, filepath)
    else:
        processed_filename, status = advanced_process_image(comm, operation, filepath)
    
    if status != "Success":
        return jsonify({"error": status}), 500
    
    return send_file(processed_filename, as_attachment=True)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "rank": rank})

if __name__ == '__main__':
    if rank == 0:
        app.run(host='0.0.0.0', port=4000)
    else:
        # Worker node logic
        while True:
            pass  # This is a placeholder for worker logic, if needed
