import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Figure 6 (NEW) — Regime curves + crossover metrics (no shaded background)
# -----------------------------

# Frequency axis
f = np.logspace(-4, 2, 3000)

# Nominal design parameters
ENOB_nom = 16
zeta_nom = 0.20

# -----------------------------
# Component models (ASD, m s^-2 / sqrt(Hz))
# -----------------------------
def Sa_brownian(f, zeta=zeta_nom):
    # Illustrative constant (consistent with previous figure set)
    return 2.0e-11 * np.ones_like(f)

def Sa_readout(f):
    # Low at ULP, rises beyond ~0.1 Hz with ~f^2 behavior
    return 5.0e-13 * np.sqrt(1.0 + (f / 0.1)**4)

def Sa_actuator(f, zeta=zeta_nom):
    # Baseline actuator/feedback noise (illustrative constant)
    return 1.0e-9 * np.ones_like(f)

def Sa_quant_clock(f):
    # Quantization + clock baseline (illustrative constant)
    return 3.0e-11 * np.ones_like(f)

def Sa_digital_shape(f, fc=5.0):
    # Digital/closed-loop shaping
    return np.sqrt(1.0 + (f / fc)**4)

def Sa_adc_from_enob(ENOB, base_at_16bit=3.0e-11):
    # +1 bit -> ~6 dB improvement (factor 2 in amplitude)
    return base_at_16bit * 2.0**(-(ENOB - 16.0))

def Sa_adc_total(f, ENOB, fc=5.0):
    return Sa_adc_from_enob(ENOB) * Sa_digital_shape(f, fc=fc)

def Sa_total(f, ENOB=ENOB_nom, zeta=zeta_nom):
    Sa_th = Sa_brownian(f, zeta)
    Sa_ro = Sa_readout(f)
    Sa_act = Sa_actuator(f, zeta)
    Sa_qc = Sa_quant_clock(f)
    Sa_adc = Sa_adc_total(f, ENOB)
    return np.sqrt(Sa_th**2 + Sa_ro**2 + Sa_act**2 + Sa_qc**2 + Sa_adc**2)

# -----------------------------
# Helper: crossover frequency where A(f)=B(f)
# First crossing from low f upward (if exists)
# -----------------------------
def crossover_frequency(f, A, B):
    D = A - B
    sgn = np.sign(D)
    change = np.where(np.diff(sgn) != 0)[0]
    if len(change) == 0:
        return np.nan
    i = change[0]
    x1, x2 = np.log10(f[i]), np.log10(f[i+1])
    y1, y2 = D[i], D[i+1]
    if y2 == y1:
        return f[i]
    x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
    return 10**x0

# -----------------------------
# Build Figure 6
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                        gridspec_kw=dict(hspace=0.14))

# =========================================================
# (a) Component curves + total (nominal ENOB, zeta)
# =========================================================
ax = axs[0]

Sa_tot = Sa_total(f, ENOB=ENOB_nom, zeta=zeta_nom)

ax.loglog(f, Sa_brownian(f, zeta_nom), lw=1.8, label="Brownian (thermal)")
ax.loglog(f, Sa_readout(f), lw=1.8, label="Readout")
ax.loglog(f, Sa_actuator(f, zeta_nom), lw=1.8, label="Actuator / feedback")
ax.loglog(f, Sa_quant_clock(f), lw=1.8, label="Quantization + clock")
ax.loglog(f, Sa_adc_total(f, ENOB_nom), lw=1.8, label=f"ADC (ENOB={ENOB_nom})")
ax.loglog(f, Sa_tot, lw=2.8, label="Total self-noise")

ax.set_xlim(1e-4, 1e2)
ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.grid(True, which="both", ls=":", lw=0.8)

# Legend: moved slightly upward (minimal offset)
ax.legend(loc="upper left", bbox_to_anchor=(0.00, 1.02), frameon=False)

# =========================================================
# (b) Crossover frequencies vs ENOB
# =========================================================
ax = axs[1]

ENOB_list = np.array([12, 14, 16, 18, 20], dtype=float)

f_cross_adc_act = []
f_cross_ro_act = []

for en in ENOB_list:
    A_adc = Sa_adc_total(f, en)
    A_act = Sa_actuator(f, zeta_nom)
    A_ro = Sa_readout(f)

    f_ca = crossover_frequency(f, A_adc, A_act)  # ADC = actuator
    f_cr = crossover_frequency(f, A_ro, A_act)   # readout = actuator

    f_cross_adc_act.append(f_ca)
    f_cross_ro_act.append(f_cr)

f_cross_adc_act = np.array(f_cross_adc_act)
f_cross_ro_act = np.array(f_cross_ro_act)

ax.semilogy(ENOB_list, f_cross_adc_act, marker="o", lw=2.5,
            label=r"$f$: ADC = actuator")
ax.semilogy(ENOB_list, f_cross_ro_act, marker="s", lw=2.0, linestyle="--",
            label=r"$f$: readout = actuator")

ax.set_xlabel("ENOB [bits]")
ax.set_ylabel("Crossover frequency [Hz]")
ax.grid(True, which="both", ls=":", lw=0.8)
ax.legend(loc="upper left", frameon=False)

# Panel labels OUTSIDE axes (non-bold)
fig.text(0.065, 0.94, "(a)", fontsize=13)
fig.text(0.065, 0.47, "(b)", fontsize=13)

plt.tight_layout(rect=[0.06, 0.04, 0.99, 0.96])
plt.savefig("Figure6_regimes_and_crossovers.png", dpi=300, bbox_inches="tight")
plt.show()
