## Schritt-für-Schritt-Pfad – Von der Idee zur laufenden Simulation (+ Zeitraffer-Clip)

> **Ziel:**  Eine 2-D-Simulation (Surge + Sway + Yaw) eines halbbalastierten Vier-Säulen-Schwimmkörpers, dessen vier 400 m-Leinen an Schleppern befestigt sind.
> Bei *t = 0 s* reißt Leine 1; wir möchten
>
> 1. die Plattformtrajektorie und
> 2. die Zugkraftverläufe der verbleibenden Leinen berechnen **und**
> 3. das Ganze als beschleunigte Top-View-Animation (Timelapse) ausgeben.

---

### 0  Werkzeug-Setup in Cursor AI

```bash
# ❶ Projekt anlegen
mkdir platform-sim && cd $_
python -m venv .venv && source .venv/bin/activate

# ❷ Bibliotheken
pip install numpy scipy matplotlib ffmpeg-python moorpy
```

*Cursor* erkennt das `venv` automatisch; Code-Completion & Inline-Erklärungen funktionieren sofort ([docs.cursor.com][1]).
*(Optional: Für hochrealistische Leinenwechselwirkungen `pip install moordyn` und OpenFAST.)* ([github.com][2], [moorpy.readthedocs.io][3])

---

### 1  Geometrie & Parameter festnageln

|    Symbol   |       Startwert | Beschreibung                              |
| :---------: | --------------: | ----------------------------------------- |
|     `m`     |   1.25 × 10⁷ kg | Gesamtmasse (halb ballastiert)            |
|    `L,B`    |           120 m | Rumpflänge & -breite (Quadrat)            |
|    `I_z`    |   8 × 10⁹ kg m² | Trägheitsmoment um *z*                    |
| `anchor[j]` |  (±60 m, ±60 m) | Ankerpunkte am Meeresboden                |
| `attach[j]` |       identisch | Befestigung am Rumpf (Eck-Säulen)         |
|     `L₀`    |           400 m | Leinenlänge                               |
|     `EA`    |     1.2 × 10⁹ N | Axiale Steifigkeit (≈ 500 MPa × 2400 mm²) |
|   `c_lin`   | 3.5 × 10⁶ N s/m | lineare Hydrodämpfung                     |

*(Passen Sie Zahlen an Ihre Plattform an; Unbekanntes können Sie später mit Sensitivitäten untersuchen.)*

---

### 2  Gleichungen in 3 DOF aufstellen

$$
\begin{aligned}
M\ddot{\xi} \;+\; C\dot{\xi} \;+\; K\xi &= F_\text{moor}(\xi) \\[2pt]
\xi &= [x,\; y,\; \psi]^{\!\top}
\end{aligned}
$$

* **M** = diag(m, m, Iₙ)
* **K** kann (für die erste Iteration) als *0* angenommen werden, weil die Hydrostatik im reinen Top-View kaum Rückstellmoment erzeugt.
* **Fₘₒₒᵣ** (Leinenkräfte) entsteht über Hooke + Geometrie:

```python
def line_force(r, a):
    vec  = r - a
    d    = np.linalg.norm(vec)
    ext  = max(0.0, d - L0)
    return -EA/L0 * ext * vec/(d+1e-9)
```

---

### 3  Code-Gerüst (Python ≥ 3.11)

```python
import numpy as np, matplotlib.pyplot as plt, matplotlib.animation as ani
from scipy.integrate import solve_ivp

# --- Konstanten (siehe Tabelle oben) ---
m, Iz = 1.25e7, 8e9
EA, L0, c_lin = 1.2e9, 400.0, 3.5e6
anchor = np.array([[ 60,  60], [-60,  60], [-60, -60], [60, -60]])
attach = anchor.copy()                    # Eck-Säulen
broken = 0                                # Leine 0 reißt

# --- RHS für ODE ---------------------------------------------------
def rhs(t, s):
    x,y,psi, dx,dy,dpsi = s
    pos  = np.array([x, y])
    R    = np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi),  np.cos(psi)]])
    F, Mz = np.zeros(2), 0.0
    for j in range(4):
        if j == broken and t >= 0:         # Bruch
            continue
        r_j = pos + R @ attach[j]
        F_j = line_force(r_j, anchor[j])
        F  += F_j
        arm = r_j - pos
        Mz += np.cross(np.append(arm,0), np.append(F_j,0))[2]
    # lineare Dämpfung
    F  -= c_lin * np.array([dx, dy])
    Mz -= c_lin * dpsi * 50
    return [dx, dy, dpsi, F[0]/m, F[1]/m, Mz/Iz]

# --- Integration ---------------------------------------------------
t_end   = 120.0          # reale Sekunden
sol = solve_ivp(rhs, (0,t_end), y0=[0,0,0,0,0,0],
                max_step=0.25, rtol=1e-6)
```

---

### 4  Timelapse-Animation (≈ 20 s Film statt 120 s Realität)

```python
# --- Daten einsammeln ---------------------------------------------
t, x, y, psi = sol.t, *sol.y[:3]
speedup = 6                      # 6× schneller als Echtzeit
fps     = 30
skip    = int(speedup / (fps*np.diff(t).mean()))

# --- Figure --------------------------------------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
plat, = ax.plot([],[], 'k', lw=2)
lines  = [ax.plot([],[], 'b')[0] for _ in range(4)]
anch   = ax.scatter(anchor[:,0], anchor[:,1], c='r', marker='x')

def init():
    plat.set_data([],[])
    for l in lines: l.set_data([],[])
    return [plat,*lines]

def update(i):
    j = i*skip
    R = np.array([[np.cos(psi[j]), -np.sin(psi[j])],
                  [np.sin(psi[j]),  np.cos(psi[j])]])
    corners = (R @ attach.T).T + np.array([x[j],y[j]])
    loop = np.vstack([corners, corners[0]])
    plat.set_data(loop[:,0], loop[:,1])
    for k,l in enumerate(lines):
        if k==broken and t[j]>=0:     # gerissene Leine unsichtbar
            l.set_data([],[])
        else:
            l.set_data([corners[k,0], anchor[k,0]],
                       [corners[k,1], anchor[k,1]])
    return [plat,*lines]

ani.FuncAnimation(fig, update, frames=len(t)//skip,
                  init_func=init, blit=True)\
   .save("timelapse.mp4", writer='ffmpeg', fps=fps)
```

`ffmpeg` sorgt für das MP4-Encoding – auf den meisten Systemen wird es automatisch mit Matplotlib erkannt.  ([matplotlib.org][4])

---

### 5  Erweiterungen & Validierung

| Wunsch                                           | Lösung                                                                                                                           |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Nichtlineare Ketten­durchhänger, Seil-Massen** | MoorDyn-C oder MoorDyn-Python koppeln (1-Zeiler: `pip install moordyn; import moordyn`); Param-Datei via YAML. ([github.com][2]) |
| **Kalibrierte Leinensteifigkeit**                | MoorPy-`Line()` → `Line.setEA(EA)` + `System.solveEquilibrium()` für Vorspannung. ([moorpy.readthedocs.io][3])                   |
| **6 DOF (Heave/Roll/Pitch)**                     | Z-DOF an ODE anhängen; Hydrodynamische Added-Mass & Dämpfung via OpenFAST-*.hydro*-Datei; gleiche Integrationsroutine.           |
| **UX-Feinschliff**                               | In Cursor per *Agent* Prompt „generate CLI flags for parameters“ hinzufügen; Auto-docs mit `pdoc`.                               |

---

### 6  Test-Checkliste

1. **Energie-Sprung**: Kontrollieren Sie, ob das Impuls-Sprunggleichgewicht unmittelbar nach dem Bruch erfüllt wird (∑T = 0 vor Bruch).
2. **Zeitschritt**: Halten Sie Δt < 0.1 · √(m / (k\_total)) → numerische Stabilität.
3. **Sensitivität**: Variieren Sie `EA`, `c_lin`; prüfen Sie Spitzenlasten.
4. **Re-Run**: Lassen Sie Cursor automatisch Tests ausführen (`pytest`) – nach jeder Codeänderung mit *⌘+⇧+Enter*.

---

### 7  Was Sie am Ende haben

* **`platform-sim/sim.py`** – vollständiger, kommentierter Solver
* **`timelapse.mp4`** – 20-Sekunden-Clip, der den Drift & die Drehung der Plattform mit sichtbar schnappender Leine zeigt
* **Parameter-JSON/YAML** – schnell für Variantenstudien austauschbar
* **Cursor-Projekt** – AI-unterstützt, um weitere Szenarien in Minuten aufzusetzen

Viel Erfolg beim Nachbauen – und falls der Timelapse zu schnell wirkt: einfach `speedup` anpassen oder `fps` erhöhen!

[1]: https://docs.cursor.com/welcome?utm_source=chatgpt.com "Cursor – Welcome to Cursor"
[2]: https://github.com/FloatingArrayDesign/MoorDyn?utm_source=chatgpt.com "FloatingArrayDesign/MoorDyn: a lumped-mass mooring ... - GitHub"
[3]: https://moorpy.readthedocs.io/?utm_source=chatgpt.com "MoorPy — MoorPy 0.9.1 documentation"
[4]: https://matplotlib.org/2.1.2/gallery/animation/basic_example_writer_sgskip.html?utm_source=chatgpt.com "Saving an animation — Matplotlib 2.1.2 documentation"
