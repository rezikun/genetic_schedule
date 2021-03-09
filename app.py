# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import dash
import dash_table
from dash.dependencies import Input, Output
import dash_html_components as html
import pandas as pd
from main import *
import dash_core_components as dcc
import pickle

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

table1 = {
    "columns": [
        {"name": ["Number"], "id": "number"},
        {"name": ["Monday"], "id": "monday"},
        {"name": ["Tuesday"], "id": "tuesday"},
        {"name": ["Wednesday"], "id": "wednesday"},
        {"name": ["Thursday"], "id": "thursday"},
        {"name": ["Friday"], "id": "friday"},
    ],
    "data": [
        {
            "number": 1,
            "monday": ['ag', 'lol'],
            "tuesday": 'mg',
            "wednesday": 'ag',
            "thursday": 'mg',
            "friday": 'oop'
        }
    ]
}

global_schedule = {}
current_week = 0
current_group = 'k-15'

number_to_day = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday"}


def reformat_table(raw_table):
    table = {}
    for group in range(len(raw_table)):
        group_name = group_to_number.inverse[group]
        weeks = []
        for week_n in range(len(raw_table[group])):
            week = {}
            for day_n in range(len(raw_table[group][week_n])):
                day_name = number_to_day[day_n]
                day = []
                for lesson_n in range(len(raw_table[group][week_n][day_n])):
                    lesson_name = subj_to_number.inverse[raw_table[group][week_n][day_n][lesson_n][0]]
                    lesson_type = "lecture" if raw_table[group][week_n][day_n][lesson_n][1] else "practice"
                    professor = raw_table[group][week_n][day_n][lesson_n][2]
                    room = raw_table[group][week_n][day_n][lesson_n][3]
                    lesson = [lesson_name,  lesson_type + " Professor: " + str(professor) + " Room: " +  str(room)]
                    day.append(lesson)
                week[day_name] = day
            weeks.append(week)
        table[group_name] = weeks
    return table


def get_week_data(raw_week):
    data = []
    for i in range(4):
        line = {'number': i + 1}
        for day in raw_week:
            if raw_week[day][i][0] == "Empty":
                line[day] = ""
            else:
                line[day] = raw_week[day][i][0] + ' ' + raw_week[day][i][1]
        data.append(line)
    return data


app.layout = html.Div([
    html.Div('Choose group', id='header-group'),
    dcc.Dropdown(
        id='group-dropdown',
        options=[
            {'label': 'K-15', 'value': 'k-15'},
            {'label': 'K-16', 'value': 'k-16'},
            {'label': 'K-17', 'value': 'k-17'},
            {'label': 'K-18', 'value': 'k-18'}
        ],
        value='k-15'
    ),
    html.Div(id='header-timestamp'),
    dash_table.DataTable(
        id='table',
        columns=[
            {"name": ["Number"], "id": "number"},
            {"name": ["Monday"], "id": "monday"},
            {"name": ["Tuesday"], "id": "tuesday"},
            {"name": ["Wednesday"], "id": "wednesday"},
            {"name": ["Thursday"], "id": "thursday"},
            {"name": ["Friday"], "id": "friday"},
        ],
        data=[],
        merge_duplicate_headers=True,
    ),
    html.Button('Previous week', id='btn-click-prev', n_clicks=0),
    html.Button('Next week', id='btn-click-next', n_clicks=0),
    html.Div(id='container-button-timestamp')
])


def update_week_number(old, by):
    global current_week
    current_week = old + by


@app.callback(Output('header-timestamp', 'children'),
              Output('table', component_property='data'),
              Input('btn-click-next', 'n_clicks'),
              Input('btn-click-prev', 'n_clicks'),
              Input('group-dropdown', 'value'))
def display(btn1, btn2, value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-click-next' in changed_id:
        update_week_number(current_week, 1)
        msg = 'Week ' + str(current_week + 1)
    elif 'btn-click-prev' in changed_id:
        update_week_number(current_week, -1)
        msg = 'Week ' + str(current_week + 1)
    else:
        global current_group
        current_group = value
        update_week_number(1, -1)
        msg = "Week 1"
    data = get_week_data(global_schedule[current_group][current_week])
    return html.Div(msg), data


if __name__ == '__main__':
    ops = GeneralOptions(read_subj_file(), read_teachers_file(), read_prof_file(), weeks=14, max_classes_per_day=4,
                         groups=["k-15", "k-16", "k-17", "k-18"], small_rooms=[x for x in range(10)],
                         lect_rooms=[x for x in range(11, 20)])
    schedule = Schedule(ops)
    #schedule.create()
    #global_schedule = reformat_table(schedule.get().schedule_table)
    with open("schedule2", "rb") as fi:
        global_schedule = reformat_table(pickle.load(fi))
    print(global_schedule)
    app.run_server(debug=True)
