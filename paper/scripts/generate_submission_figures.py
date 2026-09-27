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
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{max(size,18)}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape(label)}</text>'


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
    parts = ['<defs><pattern id="dual-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="10" stroke="#376b62" stroke-width="2" opacity="0.48"/></pattern><pattern id="neither-dots" width="10" height="10" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.2" fill="#a8756d" opacity="0.6"/></pattern></defs>']
    for i, temp in enumerate(temps):
        for j, gap in enumerate(gaps):
            r = lookup[(temp, gap)]
            color = "#d9e4ea" if int(r["friction_only_feasible"]) else "#69bda6" if int(r["dual_feasible"]) else "#f4d5cf"
            x, y = left+i*dx, top+h-(j+1)*dy
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{dx+.2:.2f}" height="{dy+.2:.2f}" fill="{color}"/>')
            pattern = "neither-dots" if not int(r["dual_feasible"]) else "dual-hatch" if not int(r["friction_only_feasible"]) else None
            if pattern:
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{dx+.2:.2f}" height="{dy+.2:.2f}" fill="url(#{pattern})"/>')
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
        parts += [line(x,top+h,x,top+h+6),text(x,top+h+27,str(t),17,"middle")]
    for gap in (15,30,45,60,75,90):
        y=top+h-(gap-gaps[0])/(gaps[-1]-gaps[0])*h
        parts += [line(left-6,y,left,y),text(left-12,y+5,str(gap),17,"end")]
    parts += [text(left+w/2,top+h+59,"Initial brake temperature (K)",18,"middle"),
              f'<text x="36" y="{top+h/2:.1f}" transform="rotate(-90 36 {top+h/2:.1f})" font-size="18" text-anchor="middle">Initial bumper gap (m)</text>',
              '<rect x="110" y="480" width="17" height="13" fill="#f4d5cf"/>',
              '<rect x="110" y="480" width="17" height="13" fill="url(#neither-dots)"/>',text(134,493,"Neither feasible",17),
              '<rect x="300" y="480" width="17" height="13" fill="#69bda6"/>',
              '<rect x="300" y="480" width="17" height="13" fill="url(#dual-hatch)"/>',text(324,493,"Dual only",17),
              '<rect x="450" y="480" width="17" height="13" fill="#d9e4ea"/>',text(474,493,"Both feasible",17),
              line(630,486,663,486,"#0b5671",3,"7 4"),text(670,493,"Friction frontier",17),
              line(630,510,663,510,"#a13b30",3),text(670,517,"Dual-brake frontier",17)]
    svg("feasibility_frontier.svg",parts,900,545)


def predictive():
    data=json.loads((DATA/"predictive_representatives.json").read_text(encoding="utf-8"))
    order=(("TRUE_EARLY_WARNING","(a) Isolated early warning"),
           ("CONSERVATIVE_WARNING_WITHOUT_BOUNDARY","(b) Warning without unique later boundary"))
    parts=[]
    for panel,(key,label) in enumerate(order):
        records=[r for r in data[key]["rows"] if r["vehicle_id"]==1]
        left,top,w,h=95,58+panel*250,680,190
        # Symmetric log compresses the large negative tail but preserves sign and zero.
        import math
        f=lambda z: math.copysign(math.log1p(abs(z)),z)
        values=[f(float(r[col])) for r in records for col in ("rho","rho_H_cert")]
        lo,hi=min(values),max(values)
        pad=max((hi-lo)*.08,.15);lo-=pad;hi+=pad
        tmax=max(float(r["time_s"]) for r in records)
        X=lambda t:left+w*t/tmax
        Y=lambda z:top+h-(f(z)-lo)/(hi-lo)*h
        parts += [text(left,top-17,label,18,weight="bold"),line(left,top,left,top+h),line(left,top+h,left+w,top+h),
                  line(left,Y(0),left+w,Y(0),"#778899",1,"5 4")]
        for tick in (0,5,10,15):
            x=X(tick);parts += [line(x,top+h,x,top+h+5),text(x,top+h+22,str(tick),17,"middle")]
        for value in (0,-1,-10,-100):
            if lo <= f(value) <= hi:
                y=Y(value);parts += [line(left-5,y,left,y),text(left-10,y+5,str(value),17,"end")]
        for field,color,dash in (("rho","#0b5671",""),("rho_H_cert","#b34f36","8 4")):
            parts.append(poly([(X(float(r["time_s"])),Y(float(r[field]))) for r in records],color,2.7,dash))
        boundary=data[key].get("boundary_time_s")
        if boundary is not None:
            x=X(float(boundary));parts += [line(x,top,x,top+h,"#6e529e",1.7,"5 4"),text(x+7,top+19,"observed boundary",17,fill="#6e529e")]
    parts += [line(110,549,143,549,"#0b5671",3),text(151,555,"Instantaneous certificate",17),
              line(425,549,458,549,"#b34f36",3,"8 4"),text(466,555,"Numerical predictive monitor",17),
              text(450,582,"Time (s); signed-log ordinate sign and zero preserved",17,"middle")]
    svg("predictive_examples.svg",parts,900,610)


def communication():
    import math
    records=rows("communication_age_metrics.csv")
    order=["zero_delay","fixed_50ms","fixed_100ms","fixed_200ms","fixed_300ms","packet_loss","burst_loss","bounded_delay","stale_messages"]
    data={r["channel"]:r for r in records}
    parts=[]
    left,top,w,h=80,55,740,270
    lower, upper = -6.5, 0.2
    Y=lambda value:top+h-(math.log10(value)-lower)/(upper-lower)*h
    for exponent in (-6,-4,-2,0):
        y=top+h-(exponent-lower)/(upper-lower)*h
        parts += [line(left,y,left+w,y,"#d6dde2",1),text(left-9,y+5,f"10^{exponent}",17,"end")]
    for j,key in enumerate(order):
        x=left+j*w/len(order)+9
        v=float(data[key]["head_mean_abs_oracle_gap"])
        y=Y(v)
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w/len(order)-18:.2f}" height="{top+h-y:.2f}" fill="#548da9"/>')
        label=key.replace("fixed_","").replace("_"," ")
        if label.endswith("ms"):
            label=label.replace("ms"," ms")
        words=label.split()
        center=x+w/len(order)/2-9
        if len(words)>1 and not label.strip().endswith("ms"):
            parts.append(text(center,top+h+19,words[0],17,"middle"))
            parts.append(text(center,top+h+37," ".join(words[1:]),17,"middle"))
        else:
            parts.append(text(center,top+h+28,label,17,"middle"))
        value_label = f"{v*1e4:.2f}e-4" if 1e-4 <= v < 1e-3 else f"{v:.3g}"
        parts.append(text(center,y-8,value_label,17,"middle"))
    parts += [line(left,top,left,top+h),line(left,top+h,left+w,top+h),
              text(left+w/2,top+h+74,"Channel condition",18,"middle"),
              text(left,32,"Head-truck mean absolute reserve error (logarithmic scale)",18),
              text(left,top+h+103,"Fixed 50 and 100 ms coincide under 100-ms sampled message consumption.",17)]
    svg("communication_age.svg",parts,900,470)


def main():
    frontier();predictive();communication()


if __name__=="__main__":
    main()
