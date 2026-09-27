"""Generate vector manuscript figures from the new, provenance-bound CSV/JSON evidence."""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/submission_revision"
OUT = ROOT / "generated/submission_revision/figures"


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def svg(name, body, width=900, height=520):
    OUT.mkdir(parents=True, exist_ok=True)
    namespace = "http:" + "//www.w3.org/2000/svg"
    text = f'<svg xmlns="{namespace}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    text += '<rect width="100%" height="100%" fill="white"/>\n'
    text += '<g font-family="DejaVu Sans,Arial,sans-serif" fill="#17212b">\n' + "\n".join(body) + "\n</g></svg>\n"
    (OUT / name).write_text(text, encoding="utf-8")


def text(x, y, label, size=16, anchor="start", weight="normal", fill="#17212b"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape(label)}</text>'


def line(x1, y1, x2, y2, color="#263746", width=1.5, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def poly(points, color, width=3, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{extra}/>'


def frontier():
    data = rows("feasibility_frontier.csv")
    temps = sorted({float(r["temperature_K"]) for r in data})
    gaps = sorted({float(r["initial_gap_m"]) for r in data})
    left, top, w, h = 105, 50, 680, 365
    dx, dy = w/len(temps), h/len(gaps)
    lookup = {(float(r["temperature_K"]), float(r["initial_gap_m"])): r for r in data}
    parts = []
    for i, temp in enumerate(temps):
        for j, gap in enumerate(gaps):
            r = lookup[(temp, gap)]
            color = "#d9e4ea" if int(r["friction_only_feasible"]) else "#69bda6" if int(r["dual_feasible"]) else "#f4d5cf"
            x, y = left+i*dx, top+h-(j+1)*dy
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{dx+.2:.2f}" height="{dy+.2:.2f}" fill="{color}"/>')
    for field, color, dash in (("friction_only_feasible", "#0b5671", "7 4"), ("dual_feasible", "#a13b30", "")):
        points = []
        for i, temp in enumerate(temps):
            yes = [gap for gap in gaps if int(lookup[(temp,gap)][field])]
            if yes and min(yes)>gaps[0]:
                points.append((left+(i+.5)*dx, top+h-(min(yes)-gaps[0])/(gaps[-1]-gaps[0])*h))
        if len(points)>1:
            parts.append(poly(points, color, 3.2, dash))
    parts += [line(left,top,left,top+h),line(left,top+h,left+w,top+h)]
    for t in (365,389,413,437,461):
        x=left+(t-temps[0])/(temps[-1]-temps[0])*w
        parts += [line(x,top+h,x,top+h+6),text(x,top+h+25,str(t),14,"middle")]
    for gap in (15,30,45,60,75,90):
        y=top+h-(gap-gaps[0])/(gaps[-1]-gaps[0])*h
        parts += [line(left-6,y,left,y),text(left-12,y+5,str(gap),14,"end")]
    parts += [text(left+w/2,top+h+59,"Initial brake temperature (K)",17,"middle"),
              text(20,top+190,"Initial bumper gap (m)",15),
              '<rect x="110" y="480" width="17" height="13" fill="#f4d5cf"/>',text(134,492,"Neither feasible",13),
              '<rect x="300" y="480" width="17" height="13" fill="#69bda6"/>',text(324,492,"Dual only",13),
              '<rect x="450" y="480" width="17" height="13" fill="#d9e4ea"/>',text(474,492,"Both feasible",13),
              line(630,486,663,486,"#0b5671",3,"7 4"),text(670,492,"Friction frontier",13),
              line(630,510,663,510,"#a13b30",3),text(670,516,"Dual-brake frontier",13)]
    svg("feasibility_frontier.svg",parts,900,545)


def predictive():
    data=json.loads((DATA/"predictive_representatives.json").read_text(encoding="utf-8"))
    order=(("TRUE_EARLY_WARNING","(a) Isolated early warning"),
           ("CONSERVATIVE_WARNING_WITHOUT_BOUNDARY","(b) Warning without unique later boundary"))
    parts=[]
    for panel,(key,label) in enumerate(order):
        records=[r for r in data[key]["rows"] if r["vehicle_id"]==1]
        left,top,w,h=95,58+panel*300,680,205
        # Symmetric log compresses the large negative tail but preserves sign and zero.
        import math
        f=lambda z: math.copysign(math.log1p(abs(z)),z)
        values=[f(float(r[col])) for r in records for col in ("rho","rho_H_cert")]
        lo,hi=min(values),max(values)
        pad=max((hi-lo)*.08,.15);lo-=pad;hi+=pad
        tmax=max(float(r["time_s"]) for r in records)
        X=lambda t:left+w*t/tmax
        Y=lambda z:top+h-(f(z)-lo)/(hi-lo)*h
        parts += [text(left,top-17,label,17,weight="bold"),line(left,top,left,top+h),line(left,top+h,left+w,top+h),
                  line(left,Y(0),left+w,Y(0),"#778899",1,"5 4")]
        for tick in (0,5,10,15):
            x=X(tick);parts += [line(x,top+h,x,top+h+5),text(x,top+h+21,str(tick),13,"middle")]
        for value in (0,-1,-10,-100):
            if lo <= f(value) <= hi:
                y=Y(value);parts += [line(left-5,y,left,y),text(left-10,y+4,str(value),12,"end")]
        for field,color in (("rho","#0b5671"),("rho_H_cert","#b34f36")):
            parts.append(poly([(X(float(r["time_s"])),Y(float(r[field]))) for r in records],color,2.7))
        boundary=data[key].get("boundary_time_s")
        if boundary is not None:
            x=X(float(boundary));parts += [line(x,top,x,top+h,"#6e529e",1.7,"5 4"),text(x+5,top+17,"observed boundary",11,fill="#6e529e")]
    parts += [line(130,619,163,619,"#0b5671",3),text(171,624,"Instantaneous certificate",13),
              line(405,619,438,619,"#b34f36",3),text(446,624,"Numerical predictive value",13),
              text(450,652,"Time (s); signed-log ordinate sign and zero preserved",13,"middle")]
    svg("predictive_examples.svg",parts,900,680)


def communication():
    import math
    records=rows("communication_age_metrics.csv")
    order=["zero_delay","fixed_50ms","fixed_100ms","fixed_200ms","fixed_300ms","packet_loss","burst_loss","bounded_delay","stale_messages"]
    data={r["channel"]:r for r in records}
    parts=[]
    left,top,w,h=80,55,740,330
    lower, upper = -6.5, 0.2
    Y=lambda value:top+h-(math.log10(value)-lower)/(upper-lower)*h
    for exponent in (-6,-4,-2,0):
        y=top+h-(exponent-lower)/(upper-lower)*h
        parts += [line(left,y,left+w,y,"#d6dde2",1),text(left-9,y+4,f"10^{exponent}",12,"end")]
    for j,key in enumerate(order):
        x=left+j*w/len(order)+9
        v=float(data[key]["head_mean_abs_oracle_gap"])
        y=Y(v)
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w/len(order)-18:.2f}" height="{top+h-y:.2f}" fill="#548da9"/>')
        parts.append(text(x+w/len(order)/2-9,top+h+20,key.replace("fixed_","").replace("_"," "),10,"middle"))
        parts.append(text(x+w/len(order)/2-9,y-7,f"{v:.3g}",10,"middle"))
    parts += [line(left,top,left,top+h),line(left,top+h,left+w,top+h),
              text(left+w/2,top+h+58,"Channel condition",17,"middle"),
              text(left,32,"Head-truck mean absolute reserve error (logarithmic scale)",17),
              text(left,top+h+85,"Fixed 50 and 100 ms coincide under 100-ms sampled message consumption.",13)]
    svg("communication_age.svg",parts,900,510)


def main():
    frontier();predictive();communication()


if __name__=="__main__":
    main()
