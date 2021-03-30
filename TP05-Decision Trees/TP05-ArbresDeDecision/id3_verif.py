#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 2.7
# Vérifier que l'arbre créé par notre ID3 est correcte
# Equipe ML (Aries & Benatchba), 2020
#
import numpy as np
import pandas as pd

def wait():
    raw_input("Appuyer sur une touche pour continuer ...")
    # input("Appuyer sur une touche pour continuer ...") # Python 3
    print("============================")

jouer = pd.read_csv("datasets/jouer.csv")
jouer.sort_values("temps", axis = 0, ascending = True, inplace = True)
print("le dataset trié par temps")
print(jouer)
print("temps=nuageux => jouer")
print("on supprime 'nuageux' et on continue")
jouer = jouer[jouer.temps != "nuageux"]
wait()

print("le dataset trié par temps et humidite")
jouer.sort_values(["temps","humidite"], axis = 0, ascending = True, inplace = True)
print(jouer)
print("temps=ensoleile et humidite=haute => ne jouer pas")
print("temps=ensoleile et humidite=normale => jouer")
print("on supprime 'ensoleile' et on continue")
jouer = jouer[jouer.temps != "ensoleile"]
wait()

print("le dataset trié par vent")
jouer.sort_values(["vent"], axis = 0, ascending = True, inplace = True)
print(jouer)
print("temps=pluvieux et pas de vent => jouer")
print("temps=pluvieux et vent => ne jouer pas")
print("Terminé")
