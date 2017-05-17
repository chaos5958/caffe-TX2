# client.py

import socket
import time
from threading import Thread
import threading
import json

HOST_IP = '127.0.0.1'
HOST_PORT = 44444
FLAG = True

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.connect((HOST_IP, HOST_PORT))

try:
    # threading._start_new_thread(sending_msg, ())
    while FLAG:
        time.sleep(1)
        #data = conn.recv(1024)
        #print repr(data)
        #time.sleep(0.5)

        #track
        msg = {
                'cmd': 'track',
                'target' : {
                    'object': 'car',
                    'index': 0,
                    }
              }

        #redetect
        msg = {
                'cmd': 'redetect'
                }

        #stop
        msg = {
                'cmd': 'stop'
                }



        output = json.dumps(msg)
        print "Send ",
        print output
        conn.send(output)
    conn.close()
except:
    print "End connection"
