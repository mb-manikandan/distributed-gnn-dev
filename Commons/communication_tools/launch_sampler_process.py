import socket
import json
from argparse import Namespace

MAX_MESSAGE=2048

class MessageSender:
    def __init__(self, addr='127.0.0.1', port=8200):
        self._addr = addr
        self._port = port

        self._socks = None
        self.connect()

        self.set_timeout(10)  # wait 10s

    def connect(self):
        self._socks = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socks.connect((self._addr, self._port)) # connect
        # print("connect")
        self._socks.setblocking(0)  # to non-blocking
        # print("connect")

    def send_message(self, msg_str):
        ret = False
        try:
            self._socks.send(msg_str.encode('utf-8'))
            ret = True
        except socket.timeout as e:
            print(e)

        return ret

    def close(self):
        if self._socks is not None:
            self._socks.close()
            del self._socks
            self._socks = None

    def __del__(self):
        self.close()

    def get_timeout(self):
        timeout = self._socks.gettimeout()
        return timeout

    def set_timeout(self, timeout):
        self._socks.settimeout(timeout)



class MessageReceiver:
    def __init__(self, addr='127.0.0.1', port=8200):
        self._addr = addr
        self._port = port

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # print(self._server_sock.gettimeout())
        self._server_sock.settimeout(10) # wait 10s
        # print(self._server_sock.gettimeout())

        self._server_sock.bind((self._addr, self._port)) # launch

        # print("connect")
        self._server_sock.listen(8) # 8 thread

        self._socks = None
        # self.accept()


    def accept(self):
        try:
        # print("connect")
            conn, addr = self._server_sock.accept()
            # print("connect")
            conn.setblocking(1)
            # print("connect")
            conn.settimeout(10) # wait 10s
            self._socks = conn
            ret = True
        except socket.timeout as e:
            ret = False

        return ret

    def receive_message(self):
        msg_str = ""
        try:
            msg_str = self._socks.recv(MAX_MESSAGE).decode('utf-8')
        except socket.timeout as e:
            msg_str = "timeout"

        return msg_str

    def receive_message_loop(self):
        while True:
            msg_str = "Error"
            try:
                msg_str = self.receive_message()
            except socket.timeout as e:
                print('Error:' + str(e))

            return msg_str

    def close(self):
        if self._socks is not None:
            self._socks.close()
            del self._socks
            self._socks = None

    def __del__(self):
        self.close()
        self._server_sock.close()

    def get_timeout(self):
        timeout = self._socks.gettimeout()
        return timeout

    def set_timeout(self, timeout):
        self._socks.settimeout(timeout)


def dumps_args_str(args):
    args_str = json.dumps(vars(args))
    return args_str


def loads_args_str(args_str):
    args_dict = json.loads(args_str)
    args = Namespace(**args_dict)
    return args
