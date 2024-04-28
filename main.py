import socket
import cv2
import os
from image_processing import process_image

HOST = '0.0.0.0'
PORT = 4000


def send_file(client_socket, filename):
    """ Send the processed file back to the client. """
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            client_socket.send(data)
    client_socket.send(b'END')

def main():
    
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print("Server listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from: {addr}")
        initial_data = client_socket.recv(1024).decode()
        op, filename = initial_data.split(',')
        processed_filename, status = process_image(op, filename)
        if processed_filename:
            send_file(client_socket, processed_filename)
        else:
            client_socket.send(status.encode())
        client_socket.close()

if __name__ == "__main__":
    main()
