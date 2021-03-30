#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy
import random

NBR = 20

# Les notes possibles
NOTES1 = numpy.arange(0, 10, 0.25)
NOTES2 = numpy.arange(13, 20, 0.25)

notes1 = numpy.concatenate((random.choices(NOTES1, k=NBR), random.choices(NOTES2, k=NBR), [5.]))

notes2 = numpy.concatenate((random.choices(NOTES1, k=NBR), random.choices(NOTES2, k=NBR), [15.]))

admis = ((notes1 + notes2)/2.0 >= 10.0).astype(int)

dataset = pandas.DataFrame({"Note1": notes1, "Note2": notes2, "Admis": admis})

#dataset["Admis"] = dataset["Admis"].map({"True": "1", "False": "0"})

dataset.to_csv("./notes.csv", index=False)
