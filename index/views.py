from django.shortcuts import render
import pandas as pd
import manage
import os
import json
# file_ = open("C:/Users/kenda/PycharmProjects/DjangoGraphJs/staticfiles/")


def covid(request):
    sözlük = {}
    file_ = open(os.path.join(manage.BASE_DIR, 'DjangoGraphJs\staticfiles\owid-covid-data.csv'))

    df = pd.read_csv(file_)
    liste = ["Turkey", "United States", "Germany", "Italy", "France"]

    for j in liste:
        df2 = df[df["location"] == j]
        df2 = df2[df.new_cases > 0]

        days = []
        for i in df2.date:
            days.append(i)
        confirmed = []
        for i in df2.total_cases:
            confirmed.append(i)
        deaths = []
        for i in df2.total_deaths:
            deaths.append(i)
        newCases = []
        for i in df2.new_cases:
            newCases.append(i)
        newDeaths = []
        for i in df2.new_deaths:
            newDeaths.append(i)
        totalTests = []
        for i in df2.total_tests:
            totalTests.append(i)

        j = j.replace(" ", "_")

        sözlük[j] = {"days": days, "confirmed": confirmed, "deaths": deaths, "newCases": newCases,
                     "newDeaths": newDeaths, "totalTests": totalTests}

    sözlük = json.dumps(sözlük)

    return render(request, "graph.html", context={"sözlük": sözlük})
