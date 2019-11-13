import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import numpy as np

#import chart_studio
#chart_studio.tools.set_credentials_file(username='cleee1994', api_key='zd2WjZZ0Eu1tSfMTusAp')

# 2 8 8 8 0.9941000000000001
# 2 10 8 10 0.99398
# 2 12 8 12 0.99391
# 3 8 8 8 0.99466
# 3 10 8 10 0.99465
# 3 12 8 12 0.9944599999999999
# 4 8 8 8 0.9946999999999999
# 4 10 8 10 0.9940300000000001
# 4 12 8 12 0.99443
# 5 8 8 8 0.99466
# 5 10 8 10 0.99368
# 5 12 8 12 0.9941899999999999
# 2 8 8 8 0.9941000000000001
# 3 8 8 8 0.99466
# 4 8 8 8 0.9946999999999999
# 5 8 8 8 0.99466
# 2 8 9 8 0.99367
# 3 8 9 8 0.99451
# 4 8 9 8 0.9940099999999997
# 5 8 9 8 0.9941800000000001
# 2 8 10 8 0.99254
# 3 8 10 8 0.99312
# 4 8 10 8 0.99321
# 5 8 10 8 0.99298
# 2 8 11 8 0.99101
# 3 8 11 8 0.9912000000000001
# 4 8 11 8 0.99192
# 5 8 11 8 0.9915199999999998
# 2 8 12 8 0.98774
# 3 8 12 8 0.9891099999999999
# 4 8 12 8 0.9894900000000002
# 5 8 12 8 0.9881500000000001

test_acc1_scale = [2, 3, 4, 5]
test_acc1 = [0.9941000000000001, 0.99466, 0.9946999999999999, 0.99466]

#energy_bits = [80.35e-2, 87.43e-2, 89.32e-2, 89.62e-2, 90.00e-2, 90.1e-2, 90.78e-2, 91.14e-2]
#testacc_bits = [20.1e-2, 38.22e-2, 47.25e-2, 57.3e-2, 65.81e-2, 72.39e-2, 77.29e-2,83.22e-2]
#current_variation = [5663.6, 5047.7, 4498.7, 4009.5, 3573.5, 3184.9, 2838.5, 2529.8]



trace_Accuracy1 = go.Scatter(
    x=test_acc1_scale,
    y=test_acc1,
    name='Test Accuracy',
    marker=dict(
        color='rgb(0,0,255)', size=20, symbol='square'
        ),
    line=dict(dash='dot', width=4),
    )
    

data = [trace_Accuracy1]
layout = go.Layout(
          autosize=False,
          width=900,
          height=500,
          
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
          
        #barmode='group',
        #bargap=0.15,

        #title=go.layout.Title(text="SparseMax vs. SoftMax Accuracy (Omniglot)", x=0.5, y=0.8),
        
        
        titlefont=dict(
                family="Helvetica",
                size=24,
                ),
        xaxis=dict(
            title='Bits used for Weights',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            range = [1.5,5.5],
            #dtick=1,
        ),
        yaxis=dict(
            title='Accuracy',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            range = [.9938, .995],
            #dtick=1,
        #    tickformat=".1f",
        ),
        legend=dict(
           x=0.05,
            y=0.1,
            bgcolor='#FFFFFF',
            bordercolor='#000000',
            borderwidth=2,
            font=dict(
                family="Helvetica",
                size=20,
                ),
        ),
)

annotations = []


plot([go.Figure(data=data, layout=layout)])
#py.iplot(fig, filename='fig1_weights')
#py.image.save_as(fig, filename='fig1_weights.png')


heat_mnist_ab = go.Heatmap(
                    z=[[0.9941000000000001, 0.99398, 0.99391],
                      [0.99466, 0.99465, 0.9944599999999999],
                      [0.9946999999999999, 0.9940300000000001, 0.99443],
                      [0.99466, 0.99368, 0.9941899999999999]
                      ],
                   x=['8b','10b', '12b'],
                   y=['2b','3b', '4b','5b'])


data = [heat_mnist_ab]
layout = go.Layout(
          autosize=False,
          width=900,
          height=500,
          
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
          
        #barmode='group',
        #bargap=0.15,

        #title=go.layout.Title(text="SparseMax vs. SoftMax Accuracy (Omniglot)", x=0.5, y=0.8),
        
        
        titlefont=dict(
                family="Helvetica",
                size=24,
                ),
        xaxis=dict(
            title='Bits used for Activations/Errors',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            #range = [7.5,10.5],
            #dtick=1,
        ),
        yaxis=dict(
            title='Bits used for Weights',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            #range = [1.5, 5.5],
            #dtick=1,
        #    tickformat=".1f",
        ),
        legend=dict(
           x=0.05,
            y=0.1,
            bgcolor='#FFFFFF',
            bordercolor='#000000',
            borderwidth=2,
            font=dict(
                family="Helvetica",
                size=20,
                ),
        ),
)

annotations = []


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='fig3_heat')
py.image.save_as(fig, filename='fig1_heat.png')













test_acc1_scale = [8, 10, 12]
test_acc1 = [0.9291600000000001, 0.9292599999999999, 0.929849999999999]

test_acc2 = [0.9351200000000002, 0.93484, 0.9355399999999999]

#energy_bits = [80.35e-2, 87.43e-2, 89.32e-2, 89.62e-2, 90.00e-2, 90.1e-2, 90.78e-2, 91.14e-2]
#testacc_bits = [20.1e-2, 38.22e-2, 47.25e-2, 57.3e-2, 65.81e-2, 72.39e-2, 77.29e-2,83.22e-2]
#current_variation = [5663.6, 5047.7, 4498.7, 4009.5, 3573.5, 3184.9, 2838.5, 2529.8]



trace_Accuracy1 = go.Scatter(
    x=test_acc1_scale,
    y=test_acc1,
    name='2 wb',
    marker=dict(
        color='rgb(0,0,255)', size=20, symbol='square'
        ),
    line=dict(dash='dot', width=4),
    )
    

trace_Accuracy2 = go.Scatter(
    x=test_acc1_scale,
    y=test_acc2,
    name='3wb',
    marker=dict(
        color='rgb(0,0,255)', size=20, symbol='square'
        ),
    line=dict(dash='dot', width=4),
    )


data = [trace_Accuracy1, trace_Accuracy2]
layout = go.Layout(
          autosize=False,
          width=900,
          height=500,
          
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
          
        #barmode='group',
        #bargap=0.15,

        #title=go.layout.Title(text="SparseMax vs. SoftMax Accuracy (Omniglot)", x=0.5, y=0.8),
        
        
        titlefont=dict(
                family="Helvetica",
                size=24,
                ),
        xaxis=dict(
            title='Bits used for Weights',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            range = [7.5, 12.5],
            #dtick=1,
        ),
        yaxis=dict(
            title='Accuracy',
            titlefont=dict(
                family="Helvetica",
                size=28,
                ),
            tickfont=dict(
                family="Helvetica",
                size=24,
            ),
            #type='log',
            range = [.925, .936],
            #dtick=1,
        #    tickformat=".1f",
        ),
        legend=dict(
           x=0.05,
            y=0.1,
            bgcolor='#FFFFFF',
            bordercolor='#000000',
            borderwidth=2,
            font=dict(
                family="Helvetica",
                size=20,
                ),
        ),
)

annotations = []


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='fig99')
py.image.save_as(fig, filename='fig99.png')

