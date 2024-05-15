# Roboter Escape Game 2



## Über das Projekt

Bei diesem Projekt geht es darum den Alphabot2 Roboter so zu programmieren, dass er in der Lage ist sich mit Hilfe seiner Line-Tracking Funktion aus einem Labyrinth zu befreien.

## Bearbeitet von:
 
 Tobias Saur, Jakob Kramer, Niklas Hart

## Vorbereitungen

- Installation der benötigten Bibliotheken (siehe requirements.txt), bei apt-get selbst installieren
- Alphabot zu Beginn möglichst zentral auf eine Linie des Labyrinths setzen

Installieren der benötigten Bibliotheken:
sudo apt-get update
sudo apt-get install ttf-wqy-zenhei
sudo apt-get install python-pip 
sudo pip install RPi.GPIO
sudo pip install spidev
sudo apt-get install python-smbus
sudo apt-get install python-serial
sudo pip install rpi_ws281x

## Ausführen des Programms

Das Hauptprogramm kann mit dem Befehl sudo python escape_game.py ausgeführt werden.
