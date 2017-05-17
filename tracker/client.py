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
        raw_data = conn.recv(4096)
        #print repr(raw_data)
        json_list = []
        json_list = raw_data.split("}\n{");
        for i in range(len(json_list)):
            print i, len(json_list)
            if i == len(json_list) - 1:
                if len(json_list) == 1:
                    parse_json(json_list[0])
                else:
                    json_list[i] = "{" + json_list[i]
                    data = json_list[i]
                    parse_json(data)
            else:
                if i == 0:
                    json_list[i] = json_list[i] + "}"
                    data = json_list[i]
                    parse_json(data)
                if i != 0:
                    json_list[i] = "{" + json_list[i] + "}"
                    data = json_list[i]
                    parse_json(data)




def parse_json(data):
    #print repr(data)
    input = json.loads(data)
    print ' '
    #print "recv msg", repr(data)
    #pprint(input)
    #print "status", input['cmd'], "data", input['target']['index']

    if input['status'].startswith("MULTI_OBJECTS"):
        print "MULTI"
    elif input['status'].startswith("NO_OBJECTS"):
        print "NO_OBJ"
    elif input['status'].startswith("SUCCESS"):
        x_min  = input['data']['x_min']
        y_min  = input['data']['y_min']
        width  = input['data']['width']
        height = input['data']['height']
        print x_min, y_min, width, height


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
