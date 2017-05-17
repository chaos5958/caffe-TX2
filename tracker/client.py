# client.py

import socket
import time
from threading import Thread
import threading
import json




HOST_IP = '127.0.0.1'
HOST_PORT = 44444
FLAG = True

def recv_msg():
    while True:
        time.sleep(1)
        data = conn.recv(1024)
        input = json.loads(data)
        print ' '
        print "recv msg", repr(data)
        #pprint(input)
        #print "cmd", input['cmd'], "data", input['target']['index']
        
        if eq(input['cmd'], "MULTI"):
            print "MULTI"
        elif eq(input['cmd'], "NO_OBJ"):
            print "NO_OBJ"
        elif eq(input['cmd'], "SUCCESS"):
            x_min  = input['data'][0]
            y_min  = input['data'][1]
            width  = input['data'][2]
            height = input['data'][3]

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.connect((HOST_IP, HOST_PORT))

try:
    threading._start_new_thread(recv_msg, ())
    while FLAG:
        # time.sleep(1)
        #data = conn.recv(1024)
        #print repr(data)
        #time.sleep(0.5)
        print "Enter command track, redetect, stop"
        command_state = 0
        while 1:
            keyboard_input = raw_input()
            if keyboard_input.startswith('track'):
                command_state = 1
                break
            elif keyboard_input.startswith('redetect'):
                command_state = 2
                break
            elif keyboard_input.startswith('stop'):
                command_state = 3
                break
            else:
                print "rewrite command"
        #track
        if(command_state == 1):
            msg = {
                    'cmd': 'track',
                    'target' : {
                        'object': 'car',
                        'index': 0,
                        }
                  }
        #redetect
        elif(command_state == 2):
            msg = {
                    'cmd': 'redetect'
                    }

        #stop
        elif(command_state == 3):
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
