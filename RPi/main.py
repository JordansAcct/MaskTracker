import sys
import time
import math
from gpiozero import RGBLED
from colorzero import Color
import RPi.GPIO as GPIO
import socket
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
udp_server_address = '10.16.230.118'
udp_port = 12000

mask_float = 0
part_mask_float = 0
no_mask_float = 0
severity_rating = 0

redPin = 17
greenPin = 27
bluePin = 22

def udp_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((udp_server_address, udp_port))

    led = RGBLED(redPin, greenPin, bluePin, active_high=True, pwm=True)
    last_value = (0,0,0)
    red_value = 0 
    green_value = 0 

    while True:
        message, address = server_socket.recvfrom(6)
        
        # process message to turn into command of some severity rating

        #cmd = input("Choose a severity rating between 0-10: ")
        mask = message.decode()[:2]
        part_mask = message.decode()[2:4]
        no_mask = message.decode()[4:6]
        mask_float = float(mask)
        part_mask_float = float(part_mask)
        no_mask_float = float(no_mask)
        if (mask_float + part_mask_float + no_mask_float == 0):
            severity_rating = 0
            cmd = 0.1
        else:
            severity_rating = round((mask_float * 0.2 + part_mask_float * 0.75 + no_mask_float) / (mask_float + part_mask_float + no_mask_float) * 100, 2)
            cmd = max(float(severity_rating) / float(100), 0.1)
        print(severity_rating)
        print(cmd)
        print("Number of masked people: " + mask)
        print("Number of partially masked people: " + part_mask)
        print("Number of no masked people: " + no_mask)
        # print("received value: " + str(cmd))
        #if (cmd < 0.5):
        #    cmd *= 10/5
        #    red_value = cmd
        #    new_value = (red_value, 1, 0)
        #elif (cmd >= 0.5):
        #    cmd = (cmd - 0.5) * (10/5)
        #    green_value = 1 - cmd
        #    new_value = (1, green_value, 0)
        
        red_value = math.log(cmd, 10) + 1
        green_value = 1 - red_value
        print(red_value)
        print(green_value)
        new_value = (red_value, green_value, 0)

        led.blink(0, 0, 1, 0, new_value, last_value, 1, background=False)
        led.value = new_value
        last_value = new_value

        socketio.emit('mask_update', {'number': mask_float})
        socketio.emit('part_mask_update', {'number': part_mask_float})
        socketio.emit('no_mask_update', {'number': no_mask_float})
        socketio.emit('number_update', {'number': severity_rating})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit('mask_update', {'number': mask_float})
    emit('part_mask_update', {'number': part_mask_float})
    emit('no_mask_update', {'number': no_mask_float})
    emit('number_update', {'number': severity_rating})

if __name__ == '__main__':
    # Start the UDP server in a separate thread
    udp_thread = Thread(target=udp_server)
    udp_thread.daemon = True
    udp_thread.start()
                                   
    # Start the Flask server
    socketio.run(app, host='0.0.0.0', port=5000)
