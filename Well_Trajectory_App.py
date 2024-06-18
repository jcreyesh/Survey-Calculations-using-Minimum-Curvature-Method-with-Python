import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as sl
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, arccos, arctan, deg2rad, rad2deg, sqrt
import altair as alt

# Image
sl.image("wt_app_2.jpg")

# Sidebar
sl.title("Well Trajectory Calculations App - Minimum Curvature Method")
sl.text("This simple App helps you to obtain a complete wellbore-trajectory using as input \ndata measure depth(md), inclination(°),\
azimuth(°) & the Vertical Section plane(°).\nThe App computes columns such as TVD, Northing, Easting, Vertical Section & DLS.\nDisplays the Vs,\
Northing vs Easting, inclination & DLS plots, as well as the 3D plot.\nFinally, you can retrieve the complete trajectory as a csv file by clicking\n\
the Download button.")

#sl.divider()

file = sl.file_uploader("Load the file")
if file is not None:
    df = pd.read_csv(file)
    sl.write(df)

col1, col2 = sl.columns([1, 1])
with col1: 
    Well = sl.text_input("Well:")
with col2:
    Vs_plane = sl.number_input("Vertical Section plane:")

# Empty containers
DLS, TVD, NS, EW, Vs = np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])
# az_vs = 60.91
md, inc, azi = df['MD'], df['INC'], df['AZI']
desp_tot, tvd_0, ns_0, ew_0 = 0, 0, 0, 0

for i in range(1,len(df)):
    # Conversiones
    I0, I1 = deg2rad(inc[i-1]), deg2rad(inc[i])
    A0, A1 = deg2rad(azi[i-1]), deg2rad(azi[i])
    MD = md[i] - md[i-1]
    
    # Displacements
    desp_par = MD * sin(I1)
    desp_tot += desp_par

    # DLS
    dls_rad =  (arccos((cos(I0) * cos(I1) + (sin(I0) * sin(I1) * cos(A1-A0))))) * (30/MD)
    dls_deg = rad2deg(dls_rad)
    DLS = np.append(DLS, dls_deg)

    # Factor F
    if dls_deg == 0:
        RF = 1
    else:
        RF = (2/dls_rad) * tan(dls_rad/2)
    
    # TVD
    tvd = ((MD/2) * (cos(I0) + cos(I1)) * RF) + TVD[i-1]
    TVD = np.append(TVD, tvd)

    # Northing - Easting
    ns = ((MD/2) * (sin(I0) * cos(A0) + sin(I1) * cos(A1)) * RF) + NS[i-1]
    # NS.append(ns)
    NS = np.append(NS, ns)
    ew = ((MD/2) * (sin(I0) * sin(A0) + sin(I1) * sin(A1)) * RF) + EW[i-1]
    EW = np.append(EW, ew)

    # Closure Distance - CD
    CD = sqrt((ew)**2 + (ns)**2)

    # Closure Azimuth - Az_CD
    # First quadrant
    if ns > 0 and ew > 0:
        az_cl = rad2deg(arctan(abs(ew/ns)))
    # Second quadrant
    elif ns < 0 and ew > 0:
        az_cl = 180 - rad2deg(arctan(abs(ew/ns)))
    # Third quadrant
    elif ns < 0 and ew < 0:
        az_cl = 180 + rad2deg(arctan(abs(ew/ns)))
    # Fourth quadrant
    else:
        az_cl = 360 - rad2deg(arctan(abs(ew/ns)))
    
    # Vertical section - Vs
    vs = cos(deg2rad(Vs_plane - az_cl)) * CD
    Vs = np.append(Vs, vs)

df['TVD'], df['NS'], df['EW'], df['VSEC'], df['DLS']  = TVD, NS, EW, Vs, DLS

# Plotting
# Create the figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=105)
ax.remove()
plt.rcParams['axes.axisbelow'] = True

# Axes_1: Vertical-Section
ax1 = plt.subplot2grid(shape=(3,2), loc=(0,0), rowspan=3)
ax1.plot(df['VSEC'], df['TVD'], color='blue')
# td = round((max(df["TVD"]) + 1000), -3)
ax1.set_ylim(round((max(df["TVD"]) + 1000), -3), 0)
# ax1.set_xlim(-600, 600)
ax1.set_xlabel(f'Vs (m) @ {Vs_plane}°',weight='bold', labelpad=8)
ax1.set_ylabel('TVD (m)',weight='bold', labelpad=8)
# ax1.set_yticks([i for i in range(0,5500,500)])
ax1.grid(c=(0.85,0.85,0.85), linestyle='dashed')

# Axes_2: Plan-View
ax2 = plt.subplot2grid(shape=(3,2),loc=(0,1)) 
ax2.plot(df['EW'],df['NS'],color='blue')
# ax2.set_xlim(-200,600)
# ax2.set_ylim(300,-600)
ax2.set_xlabel('West-East',weight='bold', labelpad=8)
ax2.set_ylabel('South-North',weight='bold', labelpad=8)
ax2.grid(c=(0.85,0.85,0.85), linestyle='dashed')

# #Axes_3: Inc-depth
ax3 = plt.subplot2grid(shape=(3,2),loc=(1,1)) 
ax3.plot(df['MD'],df['INC'],color='blue')
ax3.set_xlabel('Depth (md)',weight='bold', labelpad=8)
ax3.set_ylabel('Inc (°)',weight='bold', labelpad=8)
ax3.grid(c=(0.85,0.85,0.85), linestyle='dashed')

# Axes_4: DLS-depth
ax4 = plt.subplot2grid(shape=(3,2),loc=(2,1))
ax4.plot(df['MD'],df['DLS'],color='blue')
ax4.set_xlabel('Depth (md)',weight='bold', labelpad=8)
ax4.set_ylabel('DLS (°/30 m)',weight='bold',labelpad=8)
ax4.grid(c=(0.85,0.85,0.85), linestyle='dashed')
plt.tight_layout()
fig.suptitle(Well, fontweight="bold", y=1.02)
sl.pyplot(fig)

sl.subheader("3D-Plot")
col3, col4, col5 = sl.columns([1, 1, 1])
with col3: 
    dtick_x = sl.number_input("EW_ticks:", value=None)
with col4:
    dtick_y = sl.number_input("NS_ticks:", value=None)
with col5:
    dtick_z = sl.number_input("TVD_ticks:", value=None)
   
# 3D plot
x = df['EW']; y = df['NS']; z = df['TVD']
fig = px.line_3d(df,x,y,z, labels={'NS':'NS','EW':'EW','VD':'TVD'},
range_x=[min(x),max(x)],range_y=[min(y), max(y)],range_z=[max(z) + 500, 0])

fig.update_traces(line={'width':4,'color':'blue'})
fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=0.7, y=0.7, z=2.1),
xaxis=dict(zeroline=False,tickfont={'size':12},backgroundcolor='white',gridcolor='rgb(222,222,222)', dtick=dtick_x),
yaxis=dict(zeroline=False,tickfont={'size':12},backgroundcolor='white',gridcolor='rgb(222,222,222)', dtick=dtick_y),
zaxis=dict(zeroline=False,tickfont={'size':12},backgroundcolor='white',gridcolor='rgb(222,222,222)', dtick=dtick_z)))
fig.update_layout(width=400, height=600)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

sl.plotly_chart(fig)

# displaying the 4 precision numbers
pd.set_option("display.precision", 2)

sl.subheader("Final Trajectory")
# Exporting the dataframe
sl.write(df)

# sl.divider()
# Rounding the dataframe to 4 decimals
df = df.apply(lambda col: round(col, 4))
              
# Download Button
@sl.cache
def convert_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_csv(df)
sl.download_button(label="Download CSV", data=csv, file_name="Trajectory_" + f"{Well}" + ".csv", mime="text/csv")


