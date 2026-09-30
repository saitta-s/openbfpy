"""Mesh-refinement diagnostic for the corrected taper source at rest."""

from openbfpy import Blood, Vessel, wave_speed
from openbfpy.solver import muscl


def errors():
    blood = Blood(.004, 1060.)
    values = []
    for cells in (20, 40, 80, 160):
        v = Vessel(dict(label="taper", sn=1, tn=2, L=.01, Rp=.003, Rd=.0028,
                        E=700000., h0=.0003, M=cells), blood)
        dt = .1*v.dx/max(wave_speed(v.A, v.gamma[1:-1]))
        muscl(v, dt, blood)
        # Use the same physical central half at every mesh size; the original
        # constant-extrapolation ghost cells are not an exact tapered equilibrium.
        values.append((cells, float(max(abs(v.Q[cells//4:-cells//4]))/dt)))
    return values


if __name__ == "__main__":
    print(errors())
