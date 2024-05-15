from escape_game import decision_list

found_sequenz = False

for i in range(len(decision_list) - 1):
    for j in range(i + 1, len(decision_list)):
        if decision_list[i:j] == decision_list[j:j + j - i]:
            print("Sequenz found:", decision_list[i:j])
            found_sequenz = True
            break
    else:
        continue
    break

if found_sequenz == True:
    pass
   
# Nächste Entscheidung anders als erster Wert der Sequenz

