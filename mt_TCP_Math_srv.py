from MyServerProtocol import *
from mymath import *
import socket, threading


class ClientThread(threading.Thread):
 
    def __init__(self, ip, port, clientsocket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.csocket = clientsocket
        print("[+] New thread started for ", ip, ":", str(port))

    def run(self):
        print("Connection from : ", ip, ":", str(port))
        print("Client connected with ", ip, ":", str(port))
        clientsock.send("Welcome to the multi-thraeded server".encode())

        data = "dummydata"

        while len(data):
            data = self.csocket.recv(2048)
            print("Client(%s:%s) sent : %s" % (self.ip, str(self.port), data.decode()))
            if data.decode() == "quit":
                data = ''
            else:
                op, x, y = ParseMessage(data.decode())
                if op == "add":
                    z = add(x, y)
                if op == "sub":
                    z = sub(x, y)
                if op == "mul":
                    z = mul(x, y)
                if op == "div":
                    z = div(x, y)
                if op == "mod":
                    z = mod(x, y)
                result = BuildMessage(z)
                self.csocket.send(str.encode(result))
        self.csocket.send(str.encode("Ok By By"))
        self.csocket.close()
        print("Client at ", self.ip, " disconnected...")


host = "0.0.0.0"
port = 10000

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((host, port))

while True:
    tcpsock.listen(4)
    print("Listening for incoming connections... ")
    (clientsock, (ip, port)) = tcpsock.accept()
    # pass clientsock to the ClientThread thread object being created
    newthread = ClientThread(ip, port, clientsock)
    newthread.start()
