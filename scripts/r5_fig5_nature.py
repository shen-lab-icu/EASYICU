#!/usr/bin/env python
"""Figure 5 (publication) - cross-6-public-DB transportability of the obesity paradox.

Source: research_output/r5_obesity_crossdb/obesity_paradox_crossdb.csv
Claim: the protective BMI->mortality gradient (obesity paradox) transports in
DIRECTION across all six harmonized public ICU databases, while absolute
mortality differs ~6-fold; the extreme-obesity uptick is within CI (not
significant). A contribution only EasyICU's 6-DB concept layer can produce.
"""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.titlesize":8,"axes.labelsize":7,
 "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,"axes.spines.right":False,
 "axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
df=pd.read_csv("research_output/r5_obesity_crossdb/obesity_paradox_crossdb.csv")
CATS=["underweight","normal","overweight","obese I","obese II-III"]
XL=["under","normal","over","obeseI","obeseII-III"]
DBS=["MIMIC-III","MIMIC-IV","eICU-CRD","AmsterdamUMCdb","HiRID","SICdb"]
COL={"MIMIC-III":"#4C72B0","MIMIC-IV":"#2A4D7E","eICU-CRD":"#55A868","AmsterdamUMCdb":"#C44E52","HiRID":"#8172B3","SICdb":"#CC8800"}
x=np.arange(len(CATS))
fig,axes=plt.subplots(1,2,figsize=(7.2,3.4))
# Panel A: absolute mortality + CI
axA=axes[0]
for db in DBS:
    s=df[df.database==db].set_index("bmi_category").reindex(CATS)
    y=s.mortality_pct.values; lo=s.ci_lo.values; hi=s.ci_hi.values
    axA.plot(x,y,"-o",ms=2.6,lw=1.1,color=COL[db],label=db)
    axA.fill_between(x,lo,hi,color=COL[db],alpha=0.12,linewidth=0)
axA.set_xticks(x); axA.set_xticklabels(XL,rotation=30,ha="right")
axA.set_ylabel("In-hospital mortality (%)")
axA.set_title("a  Absolute mortality by BMI (6 public DBs)",loc="left",fontsize=8)
axA.legend(loc="upper right",ncol=1,fontsize=5.6)
axA.text(0.02,0.02,"~6x between-DB spread",transform=axA.transAxes,fontsize=5.8,color="0.3")
# Panel B: relative to normal-BMI (direction transportability)
axB=axes[1]
for db in DBS:
    s=df[df.database==db].set_index("bmi_category").reindex(CATS)
    ref=s.loc["normal","mortality_pct"]
    rr=s.mortality_pct.values/ref
    axB.plot(x,rr,"-o",ms=2.6,lw=1.1,color=COL[db],label=db)
axB.axhline(1.0,color="0.5",lw=0.8,ls="--")
axB.set_xticks(x); axB.set_xticklabels(XL,rotation=30,ha="right")
axB.set_ylabel("Mortality relative to normal BMI")
axB.set_title("b  Direction transports across all DBs",loc="left",fontsize=8)
axB.text(0.02,0.04,"overweight/obese < 1 in all 6 DBs\n(protective gradient is robust)",
         transform=axB.transAxes,fontsize=5.8,color="0.25")
fig.suptitle("The obesity paradox transports in direction across six harmonized public ICU databases, "
             "but absolute mortality does not",fontsize=8.5,fontweight="bold",y=1.02)
fig.tight_layout()
out=Path("research_output/r5_obesity_crossdb/Figure5")
for ext,kw in [(".svg",{}),(".pdf",{}),(".tiff",{"dpi":600,"pil_kwargs":{"compression":"tiff_lzw"}}),(".png",{"dpi":300})]:
    fig.savefig(str(out)+ext,bbox_inches="tight",**kw)
print("wrote",out,"svg/pdf/tiff/png")
