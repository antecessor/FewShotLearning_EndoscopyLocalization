import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np

def CreateGanttChartForPosition(label, time, dist):
    df = []
    for i in range(len(label) - 1):
        if dist[i] < 0.25:
            resource = "high confidence"
        else:
            if label[i] != "Other":
                resource = "suspect"
            else:
                resource = "high confidence"
        mon, sec = divmod(time[i], 60)
        hr, mon = divmod(mon, 60)
        startTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        mon, sec = divmod(time[i + 1], 60)
        hr, mon = divmod(mon, 60)
        endTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        df.append(dict(Task=label[i], Start=startTime, Finish=endTime, Resource=resource))

    colors = {'high confidence': 'rgb(50, 200, 50)',
              'suspect': 'rgb(200, 100, 100)'}

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)
    fig.show()


def CreateGanttChartForAnomaly(label, time, labelNames):
    df = []
    for i in range(len(label) - 1):
        resource = label[i]

        mon, sec = divmod(time[i], 60)
        hr, mon = divmod(mon, 60)
        startTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        mon, sec = divmod(time[i + 1], 60)
        hr, mon = divmod(mon, 60)
        endTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        df.append(dict(Task=label[i], Start=startTime, Finish=endTime, Resource=resource))

    colors = {labelNames[0]: 'rgb(150, 50, 100)',
              labelNames[1]: 'rgb(50, 200, 100)',
              labelNames[2]: 'rgb(50, 100, 200)',
              labelNames[3]: 'rgb(200, 60, 50)'}

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)

    fig.show()


def CreateGanttChartForAnomalyAndPosition(labelAnomaly, timeAnomaly, labelNamesAnomaly, labelPosition, timePosition, dist, labelNamesPosition):
    df = []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for i in range(len(labelAnomaly) - 1):
        resource = labelAnomaly[i]

        mon, sec = divmod(timeAnomaly[i], 60)
        hr, mon = divmod(mon, 60)
        startTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        mon, sec = divmod(timeAnomaly[i + 1], 60)
        hr, mon = divmod(mon, 60)
        endTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        df.append(dict(Task=labelAnomaly[i], Start=startTime, Finish=endTime, Resource=resource))

    colors = {labelNamesAnomaly[0]: 'rgb(150, 50, 100)',
              labelNamesAnomaly[1]: 'rgb(50, 200, 100)',
              labelNamesAnomaly[2]: 'rgb(50, 100, 200)',
              labelNamesAnomaly[3]: 'rgb(200, 60, 50)'}

    figs = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                           group_tasks=True)

    df = []
    for i in range(len(labelPosition) - 1):
        # if dist[i] < 0.25:
        #     resource = "high confidence"
        # else:
        #     if labelPosition[i] != "Other":
        #         resource = "suspect"
        #     else:
        #         resource = "high confidence"
        resource = labelPosition[i]
        mon, sec = divmod(timePosition[i], 60)
        hr, mon = divmod(mon, 60)
        startTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        mon, sec = divmod(timePosition[i + 1], 60)
        hr, mon = divmod(mon, 60)
        endTime = "2016-01-01 %d:%02d:%02d" % (hr, mon, sec)

        df.append(dict(Task=labelPosition[i], Start=startTime, Finish=endTime, Resource=resource))

    # colors = {'high confidence': 'rgb(50, 200, 50)',
    #           'suspect': 'rgb(200, 100, 100)'}
    labelNamesPosition.append("Other")
    colors = {item: "rgb(100, {0}, {0})".format(int(10 * np.where(np.asarray(labelNamesPosition) == item)[0])) for ind, item in enumerate(labelPosition)}

    figs2 = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True, show_hover_fill=True)

    for trace in figs.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in figs2.data:
        fig.add_trace(trace, row=2, col=1)

    fig.show()
