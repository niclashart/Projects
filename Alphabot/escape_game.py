import RPi.GPIO as GPIO
from AlphaBot2 import AlphaBot2
from rpi_ws281x import Adafruit_NeoPixel, Color
from TRSensors import TRSensor
import time
import sys
from erkennung import erkennen

Button = 7

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Button, GPIO.IN, GPIO.PUD_UP)

# LED strip configuration:
LED_COUNT = 4      # Number of LED pixels.
LED_PIN = 18      # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 5       # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 255     # Set to 0 for darkest and 255 for brightest
# True to invert the signal (when using NPN transistor level shift)
LED_INVERT = False
red = Color(100, 0, 0)
blue = Color(0, 100, 0)
green = Color(0, 0, 100)
yellow = Color(100, 100, 0)
black = Color(0, 0, 0)

# Create NeoPixel object with appropriate configuration.
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ,
                          LED_DMA, LED_INVERT, LED_BRIGHTNESS)

# Intialize the library (must be called once before other functions).
strip.begin()

# Konstruktor für die Line-Tracking Funktionalität
TR = TRSensor()

# Kontruktor für den Alphabot um die Fahrmanöver auszuführen
Ab = AlphaBot2()
Ab.stop()

print("Watch me going through that maze!")
time.sleep(0.5)

# Kalibrierung: Roboter dreht nach links und rechts, sodass jeder Sensor die weiße und schwarze Fläche "sieht"
for i in range(0, 100):
    if (i < 25 or i >= 75):
        Ab.right()
        Ab.setPWMA(25)
        Ab.setPWMB(20)
    else:
        Ab.left()
        Ab.setPWMA(25)
        Ab.setPWMB(20)
    TR.calibrate()
Ab.stop()

print(TR.calibratedMin)
print(TR.calibratedMax)

# Bis der zentrale Joystick gedrückt wird, können Position und die einzelnen Sensorwerte ausgegeben werden
while (GPIO.input(Button) != 0):
    position, Sensors = TR.readLine()
    print(
        f"Position: {int(position):4},    {int(Sensors[0]):4}, {int(Sensors[1]):4}, {int(Sensors[2]):4}, {int(Sensors[3]):4}, {int(Sensors[4]):4}")
    time.sleep(0.05)
time.sleep(0.5)

# Roboter beginnt vorwärts zu fahren und durch das Labyrinth zu navigieren
Ab.forward()

while (GPIO.input(Button) != 0):
    position, Sensors = TR.readLine()
    print(
        f"Position: {int(position):4},    {int(Sensors[0]):3}, {int(Sensors[1]):3}, {int(Sensors[2]):3}, {int(Sensors[3]):3}, {int(Sensors[4]):3}")

    higher = 100    # Schwellwert für schwarze Linie

    poss = erkennen(Sensors, higher)

    # Falls abbiegen (links oder rechts) möglich ist
    if poss[1] == True or poss[2] == True:

        # Falls abbiegen (links und rechts) möglich
        if poss[1] == True or poss[2] == True:
            Ab.stop()
            time.sleep(0.5)
            Ab.moveoverline()
            _, new_values = TR.readLine()
            if new_values[2] > higher:
                poss[0] = True
            else:
                poss[0] = False

    print(poss)

    # Im folgenden werden die von den Sensoren erkannten Möglichkeiten behandelt
    # Entscheidungen werden in Liste als int Werte abgespeichert. Umgedreht: 0, Links: 1, Rechts: 2
    decision_list = []

    # 1. Fall: Nur umdrehen möglich (Sackgasse) -> funktioniert schlecht deswegen auskommentiert
    # if poss == [False, False, False]:
    #     Ab.stop()

    #     for i in range(0, LED_COUNT):
    #         strip.setPixelColor(i, red)
    #     strip.show()

    #     time.sleep(0.5)
    #     Ab.turnaround()
    # decision_list.append(0)
    #     print(decision_list)

    # 2. Fall: Links und rechts möglich (geradeaus nicht)
    if poss == [False, True, True]:
        Ab.stop()

        strip.setPixelColor(0, red)
        strip.setPixelColor(3, red)
        strip.show()

        time.sleep(0.5)
        Ab.turnleft()
    #   decision_list.append(1)
    #   print(decision_list)

    # 3. Fall: Nur Links möglich
    if poss == [False, True, False]:
        Ab.stop()

        strip.setPixelColor(3, red)
        strip.show()

        time.sleep(0.5)
        Ab.turnleft()
    # decision_list.append(1)
    #   print(decision_list)

    # 4. Fall: Nur rechts möglich
    if poss == [False, False, True]:
        Ab.stop()

        strip.setPixelColor(0, red)
        strip.show()

        time.sleep(0.5)
        Ab.turnright()
    #   decision_list.append(2)
    #   print(decision_list)

    Ab.follow_line(position)
    Ab.forward()

    for i in range(0, LED_COUNT):
        strip.setPixelColor(i, black)
        strip.show()


print("Programm durch drücken des zentralen Joystick-Buttons beendet")
for i in range(0, LED_COUNT):
    strip.setPixelColor(i, black)
    strip.show()
strip._cleanup()
sys.exit()
