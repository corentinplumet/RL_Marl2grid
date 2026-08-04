import plotly.graph_objects as go
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
try:
    fig.write_image("test_plotly.png")
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
