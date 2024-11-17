#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:40:43 2024

@author: calebkumar
"""

import numpy as np
import plotly.graph_objects as go

# Generate sample data for two overlapping distributions
x = np.linspace(84, 98, 1000)
dist1 = np.exp(-((x - 87) ** 2) / (2 * 1.5 ** 2)) / (1.5 * np.sqrt(2 * np.pi))
dist2 = np.exp(-((x - 92) ** 2) / (2 * 2 ** 2)) / (2 * np.sqrt(2 * np.pi))

# Normalize the distributions for easier visualization
dist1 = dist1 / max(dist1)
dist2 = dist2 / max(dist2)

# Create the Plotly figure
fig = go.Figure()

# Add the first distribution
fig.add_trace(go.Scatter(
    x=x,
    y=dist1,
    fill='tozeroy',
    name="Distribution 1",
    line=dict(color='blue'),
    opacity=0.5
))

# Add the second distribution
fig.add_trace(go.Scatter(
    x=x,
    y=dist2,
    fill='tozeroy',
    name="Distribution 2",
    line=dict(color='lightblue'),
    opacity=0.5
))

# Add a vertical line at a specific value
fig.add_vline(
    x=92,
    line=dict(color="black", dash="dash"),
    annotation_text="92%",
    annotation_position="top right"
)

# Customize layout
fig.update_layout(
    title="Interactive Visualization of Overlapping Distributions",
    xaxis_title="Arterial oxygen saturation at pulse oximeter reading of 92%",
    yaxis_title="Density",
    template="plotly_white",
)

# Show the figure
fig.show()
fig.write_html('test.html')