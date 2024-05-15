# Entscheidung anhand eines Schwellwerts wohin gefahren werden kann:

# poss[0]: Geradeaus möglich?
# poss[1]: Links möglich?
# poss[2]: Rechts möglich?

def erkennen(Sensors, higher):
    poss = [False, False, False]
    if Sensors[2] >= higher or Sensors[1] >= higher or Sensors[3] >= higher:
        poss[0] = True

    if Sensors[0] >= higher:
        poss[1] = True

    if Sensors[4] >= higher:
        poss[2] = True
    return poss
