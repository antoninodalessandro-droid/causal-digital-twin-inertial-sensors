#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 (SRL) — Open-loop vs closed-loop transfer functions for an ULP
force-feedback seismometer (digital twin, no experimental data required).

Final refinements:
- Legend in panel (b) moved downward (manual placement)
- Panel labels (a), (b) moved further upward, fully outside axes
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main(outdir="outputs", basename="Figure2_open_closed_loop_transfer"):
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # Representative ULP parameters
    # -----------------------------
    m = 1.0
    T0 = 120.0
    f0 = 1.0 / T0
    w0 = 2.0 * np.pi * f0
    zeta = 0.20

    k = m * w0**2
    c = 2.0 * zeta * m * w0

    Kp = 200.0 * k
    Kd = 50.0 * c

    # Frequency axis
    f = np.logspace(-4, 2, 4000)
    w = 2.0 * np.pi * f
    s = 1j * w

    # Open-loop
    denom_ol = (k - m * w**2) + 1j * (c * w)
    Hx_ol = (-m) / denom_ol

    # Closed-loop
    denom_cl = (k + Kp - m * w**2) + 1j * ((c + Kd) * w)
    Hx_cl = (-m) / denom_cl

    HF_cl = (Kp + s * Kd) * Hx_cl
    HF_ideal = m * np.ones_like(f)

    mag_Hx_ol = np.abs(Hx_ol)
    mag_Hx_cl = np.abs(Hx_cl)
    mag_HF_cl = np.abs(HF_cl)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(11.5, 7.0), dpi=150)

    ax1 = fig.add_axes([0.10, 0.56, 0.86, 0.38])
    ax2 = fig.add_axes([0.10, 0.10, 0.86, 0.38])

    # ---- (a) Compliance
    ax1.loglog(f, mag_Hx_ol, label="open-loop")
    ax1.loglog(f, mag_Hx_cl, label="closed-loop (force-feedback)")
    ax1.axvline(f0, linewidth=1.0)

    ax1.set_xlim(1e-4, 1e2)
    ax1.set_ylabel(r"$|X(f)/\ddot{u}(f)|\;[\mathrm{s}^2]$")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax1.legend(loc="upper right", frameon=False)

    # f0 annotation (shifted right)
    idx = np.argmin(np.abs(f - f0))
    ax1.text(
        f0 * 1.35,
        mag_Hx_ol[idx] * 0.9,
        r"$f_0 = 1/T_0$",
        fontsize=10,
        ha="left",
        va="center"
    )

    # ---- (b) Force balance
    ax2.loglog(f, mag_HF_cl, label=r"closed-loop: $|F_{\mathrm{fb}}/\ddot{u}|$")
    ax2.loglog(f, HF_ideal, "--", label=r"ideal: $m$")

    ax2.set_xlim(1e-4, 1e2)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"$|F_{\mathrm{fb}}(f)/\ddot{u}(f)|\;[\mathrm{kg}]$")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.6)

    # Legend moved downward (manual)
    ax2.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.85),
        frameon=False
    )

    # Parameters
    ax2.text(
        0.02, 0.08,
        r"$T_0=120\,\mathrm{s},\ \zeta=0.20,\ m=1.0\,\mathrm{kg}$" "\n"
        r"$K_p=200\,k,\ K_d=50\,c$",
        transform=ax2.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom"
    )

    # ---- Panel labels (figure coordinates, higher)
    fig.text(0.095, 0.975, "(a)", fontsize=12, ha="left", va="top")
    fig.text(0.095, 0.515, "(b)", fontsize=12, ha="left", va="top")

    # Save
    fig.savefig(f"{outdir}/{basename}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{outdir}/{basename}.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)

    print("Figure 2 regenerated with corrected legend and panel labels.")


if __name__ == "__main__":
    main()
