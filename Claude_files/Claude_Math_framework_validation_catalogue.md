# A Rigorous Verification & Validation Test Catalogue for a Relativistic Boris Runaway-Electron Code

## TL;DR
- Build a six-tier verification ladder (T0 pure numerics → T6 published RE benchmarks); the single most decisive early test is a crossed-**E×B** drift case at Lorentz factor γ≳10, which *passes* for the Vay and Higuera–Cary pushers but *fails* for standard relativistic Boris — the canonical discriminator among relativistic integrators. Ripperda et al. (ApJS 235, 21, 2018) found "the error for the Boris scheme increases dramatically when increasing [γ], while for the other methods the growth is much smaller … the error affecting the Boris scheme increases approximately one order of magnitude per increasing time step, whereas for the other methods the error does not depend much on the time step."
- Every tier has a closed-form reference: the matrix exponential exp(qFΔt/m) for arbitrary uniform (E,B) fields (machine-precision at any γ), the canonical toroidal momentum p_φ = γmRv_φ + qψ conserved exactly in axisymmetric tokamak fields, and the Decker/Hirvijoki (p,ξ) momentum-space attractor for the electric-acceleration / collisional-drag / synchrotron-radiation balance.
- Boris is second-order, phase-space-volume-preserving, energy-conserving in pure B, but **not symplectic** and **not E×B-preserving** relativistically; expect global error ∝Δt², gyrophase error ∝(ω_cΔt)², bounded (non-secular) energy error, and a round-off floor near machine epsilon that a 4th-order Runge–Kutta reference lacks (RK4 shows secular energy drift — the canonical demonstration figure).

## Key Findings

1. **Standard relativistic Boris fails the relativistic E×B drift test.** Vay (Phys. Plasmas 15, 056701, 2008) showed the Boris velocity-average decouples the electric and magnetic advances, breaking Lorentz invariance and introducing a spurious drift error that grows with γ. Vay and Higuera–Cary (Phys. Plasmas 24, 052104, 2017) fix this. **Higuera–Cary is the recommended production pusher for an RE code** because, in the authors' own words, "The Boris method and the current method are shown to be volume-preserving, while the method of [Vay] and the current method preserve the E×B velocity. Thus, of these second-order relativistic momentum integrations, only the integrator introduced here both preserves volume and gives the correct E×B velocity."
2. **The most powerful long-time tokamak diagnostic is the canonical toroidal momentum p_φ**, conserved exactly in any static axisymmetric field. Its conservation level (relative drift over ~10⁶ toroidal transits) is a more sensitive integrator-quality metric than energy for a tokamak orbit code.
3. **Runaway physics has clean single-particle attractors.** With a parallel E-field, relativistic collisional drag, and synchrotron radiation reaction, the (p,ξ) equations possess a stable O-point (attractor) that serves as an exact fixed-point validation test; the two-threshold-field theory (Aleynikov & Breizman, Phys. Rev. Lett. 114, 155001, 2015 — "Two different threshold electric fields characterize a minimal field required for sustainment of the existing runaway population and a higher field required for the avalanche onset") gives a closed-form effective critical field E₀.
4. **Stochastic collision operators degrade the convergence order.** Euler–Maruyama is strong-order 1/2, weak-order 1; Milstein restores strong-order 1 (Dimits et al., only if the area-integral terms are retained). Ensemble statistical error scales as 1/√N. These must be separated from the deterministic pusher's order-2 behaviour.

## Details

### Notation and sign conventions (SI throughout)
- Charge q (electron q = −e, e = 1.602176634×10⁻¹⁹ C), mass m (electron m_e = 9.1093837015×10⁻³¹ kg), c = 2.99792458×10⁸ m/s, ε₀ = 8.8541878128×10⁻¹² F/m.
- Proper velocity (spatial part of the 4-velocity) **u** = γ**v**, with γ = √(1 + |u|²/c²). Momentum **p** = m**u** = γm**v**.
- Equations of motion: d**u**/dt = (q/m)(**E** + **v**×**B**), d**x**/dt = **v** = **u**/γ.
- Gyrofrequency (signed) Ω_c = qB/(γm); non-relativistic ω_c = qB/m. Larmor radius r_L = γm v_⊥/(|q|B) = p_⊥/(|q|B).

### Part A — The relativistic Boris algorithm (exact derivation)
The leapfrog stagger places **x** on integer steps and **u** on half-integer steps (or, in the "synchronized" variant, both are centred on integer steps via two half-position updates). The velocity update splits into half-electric-push / magnetic-rotation / half-electric-push:

1. First half electric impulse: **u⁻** = **uⁿ** + (qΔt/2m)**E**(x^{n+1/2}).
2. Rotation: γ⁻ = √(1 + |u⁻|²/c²); **t** = **B** qΔt/(2mγ⁻); **s** = 2**t**/(1 + |t|²); **u′** = **u⁻** + **u⁻**×**t**; **u⁺** = **u⁻** + **u′**×**s**.
3. Second half electric impulse: **uⁿ⁺¹** = **u⁺** + (qΔt/2m)**E**(x^{n+1/2}).
- Position: **xⁿ⁺¹** = **x^{n+1/2}** + (Δt/2)(**uⁿ⁺¹**/γⁿ⁺¹).

**Tangent-of-half-angle rotation:** the operation implements an exact rotation by angle θ_B where tan(θ_B/2) = |t| = |q|BΔt/(2γ⁻m). Because a rotation preserves |u⁺| = |u⁻|, γ is unchanged by the magnetic step, so **the Boris rotation does exactly zero work** — this is why Boris conserves energy to machine precision in pure-B fields.

**Half-step "kickback" / initialization:** because **u** lives at half-steps, the physical initial condition u(0) must be pushed back a half step to u(−Δt/2) (a half-step backward rotation plus half E-impulse). Skipping this yields an O(Δt) phase/energy offset that pollutes convergence studies — a frequent bug caught by the reversibility test (T0.4).

**Known properties (with citations):**
- Second-order global accuracy in Δt (local truncation error third order): Boris (1970); Birdsall & Langdon, *Plasma Physics via Computer Simulation*.
- **Phase-space volume preservation** with det(∂ψ/∂z) = 1, yet **not symplectic**: Qin, Zhang, Xiao, Liu, Sun & Tang, Phys. Plasmas 20, 084503 (2013) — "We show that the Boris algorithm conserves phase space volume, even though it is not symplectic. The global bound on energy error typically associated with symplectic algorithms still holds for the Boris algorithm." This yields a bounded (non-secular) energy error, the hallmark demonstration versus RK4.
- Time reversibility (the map is time-symmetric).
- **Gyrophase error:** the numerical rotation angle satisfies tan(θ_B/2) = ω_cΔt/2 rather than θ = ω_cΔt, giving a per-step phase lag δθ ≈ −(1/12)(ω_cΔt)³ and an accumulated gyrophase error that scales as (ω_cΔt)². Zenitani & Umeda (Phys. Plasmas 25, 112110, 2018) express the accumulated error as δθ ≈ (π/√2)(ω_cΔt/n)² for n sub-steps. Their **exact-gyration / gyrophase-corrected variant** replaces θ_B by the exact θ_C via a Rodrigues/Euler rotation, u⁺ = u_∥ + (u − u_∥)cosθ + (u×b̂)sinθ with θ = (qB/γm)Δt, eliminating the gyrophase error entirely while retaining volume preservation.
- **E×B drift error:** standard Boris does not reproduce the relativistic E×B drift; the error grows with γ (Vay 2008; PICsar/Belyaev et al. quantify that "for ω_cΔt ≳ 2 the Boris mover starts to deviate significantly from the analytical solution with increasing ω_cΔt, whereas the Vay mover still [captures the drift]").

### Part B — Vay (2008) and Higuera–Cary (2017) update formulas
All three schemes share the split structure and differ only in the mean velocity v̄ used in the magnetic rotation (Ripperda et al. 2018, §2, give the algebra used below verbatim).

**Boris:** v̄ = (uⁿ⁺¹ + uⁿ)/(2γ^{n+1/2}), with γ^{n+1/2} = γ⁻ computed from u⁻ before rotation. Volume-preserving; not E×B-preserving.

**Vay:** v̄ = ½(uⁿ/γⁿ + uⁿ⁺¹/γⁿ⁺¹). Steps: u^{n+1/2} = uⁿ + (qΔt/2m)(E + (uⁿ/γⁿ)×B); u′ = u^{n+1/2} + (qΔt/2m)E; τ = (qΔt/2m)B; u* = u′·τ/c; γ′ = √(1 + u′²/c²); σ = γ′² − τ²; γⁿ⁺¹ = √{[σ + √(σ² + 4(τ² + u*²))]/2}; t = τ/γⁿ⁺¹; s = 1/(1 + |t|²); uⁿ⁺¹ = s[u′ + (u′·t)t + u′×t]. Lorentz-invariant / E×B-preserving; **not** volume-preserving.

**Higuera–Cary:** v̄ = (uⁿ⁺¹ + uⁿ)/(2γ̄), γ̄ = √(1 + ((uⁿ⁺¹ + uⁿ)/2c)²). Steps: u⁻ = uⁿ + (qΔt/2m)E; γ⁻ = √(1 + u⁻²/c²); τ = (qΔt/2m)B; u* = u⁻·τ/c; σ = γ⁻² − τ²; γ⁺ = √{[σ + √(σ² + 4(τ² + u*²))]/2}; t = τ/γ⁺; s = 1/(1 + |t|²); u⁺ = s[u⁻ + (u⁻·t)t + u⁻×t]; uⁿ⁺¹ = u⁺ + (qΔt/2m)E + u⁺×t. **Both volume-preserving and E×B-preserving — the recommended default.**

### Part C — Exact analytic reference solutions
- **Uniform B (relativistic gyration):** |u| and γ constant; circular perpendicular motion with Ω_c = qB/(γm), r_L = γmv_⊥/(|q|B); parallel velocity unchanged (helix).
- **Uniform E (parallel acceleration / hyperbolic motion):** for E along x, the proper velocity is exactly linear in lab time, u_x(t) = u_{x0} + (qE/m)t, with γ(t) = √(1 + u²/c²) and x(t) = x₀ + (mc²/qE)[√(1 + (u_{x0} + qEt/m)²/c²) − √(1 + u_{x0}²/c²)]. Test at γ = 10²–10³ for runaway relevance.
- **E parallel to B:** parallel and perpendicular dynamics decouple; helical path with linearly increasing parallel proper velocity; μ evolves as γ changes.
- **Crossed E⊥B:** if B² > E²/c² (magnetically dominated), boost to the drift frame v_d = c²(E×B)/B² (|v_d| < c) where E′ = 0 and the particle purely gyrates about B′∥B; the lab motion is gyration + drift and the time-averaged E·v = 0 (no secular energy gain). Non-relativistic limit v_d = (E×B)/B². If E²/c² > B² (electrically dominated), boost to v_d = c²(E×B)/E² where B′ = 0 and the particle is linearly accelerated along E′, gaining energy without bound (Takeuchi, "Relativistic E×B acceleration," Phys. Rev. E 66, 037402, 2002). The relativistic drift acquires an additional component perpendicular to both B and v_d (AIP Advances 11, 035303, 2021).
- **General constant field — matrix exponential (machine-precision reference):** the covariant EoM du^μ/dτ = (q/m)F^μ_ν u^ν has the exact solution u(τ) = exp[(q/m)F τ] u(0), F the electromagnetic field tensor. By Cayley–Hamilton, F satisfies a characteristic polynomial fixed by the two invariants I₁ = E²/c² − B² and I₂ = E·B/c, so the 4×4 matrix exponential is closed-form; map lab time to proper time via dt = γ dτ. This is the gold-standard reference for arbitrary uniform (E,B) at any γ (Pétri, arXiv:1612.04563, uses exactly this to validate to γ ≈ 10⁵ with 8-digit agreement in the drift-frame γ).

### Part D — Tokamak field models and invariants
- **Large-aspect-ratio vacuum toroidal field:** B_φ = B₀R₀/R (along e_φ) plus poloidal B_θ; the 1/R dependence is exact and directly testable.
- **Circular cross-section with q-profile:** B_θ(r) = r B₀/(R₀ q(r)); safety factor q(r) = r B_φ/(R₀ B_θ).
- **Solov'ev analytic Grad–Shafranov equilibrium:** GS operator Δ*ψ = R ∂_R(R⁻¹ ∂_R ψ) + ∂_ZZ ψ. The Solov'ev closure μ₀p′(ψ) = const, FF′(ψ) = const gives a closed-form polynomial ψ(R,Z); e.g. the normalized BORAY form ψ = (ψ₀/R₀⁴){(R²−R₀²)² + (Z²/E²)(R²−R_x²) − τR₀²[R²ln(R²/R₀²) − (R²−R₀²) − (R²−R₀²)²/(2R₀²)]}, or the Cerfon–Freidberg "one size fits all" 12-coefficient homogeneous-plus-particular solution (Phys. Plasmas 17, 032502, 2010). Because ψ is analytic, B = ∇ψ×∇φ + F∇φ can be verified against the closed form to machine precision, decoupling field-evaluation bugs from pusher bugs.
- **Canonical toroidal momentum (exact relativistic):** p_φ = γm R v_φ + qψ = m R u_φ + qψ (ψ the poloidal flux). Conserved exactly in any static axisymmetric field (Rome, ORNL/TM-6352, gives P_φ = γmRv_φ − Zeψ/c in Gaussian form; the SI sign depends on the ψ definition). **Numerical subtlety:** with staggered leapfrog v_φ is at half-steps and ψ(x) at integer steps, so compute p_φ by time-centring (average adjacent half-step velocities, or use the synchronized integer-step state) to avoid a spurious O(Δt) oscillation masking the true drift. Expect bounded relative conservation |Δp_φ/p_φ| ~ 10⁻⁶–10⁻¹⁰ (Δt-dependent), with no secular drift over 10⁶ transits.
- **Energy:** conserved in static B plus conservative E; with an inductive toroidal E the work done equals ∮ qE·dl exactly.
- **Magnetic moment μ = p_⊥²/(2mB):** adiabatic invariant, conserved to exponential accuracy when ρ/L ≪ 1.
- **Orbit topology:** trapping condition v_∥/v < √(2ε/(1+ε)) with ε = r/R the inverse aspect ratio; banana width Δ_b ~ q ρ_θ/√ε ~ (q/√ε)(mv_∥/qB_θ); bounce frequency ω_b ≈ (√ε/qR)(v_∥/√2) for deeply trapped particles; banana orbits do not enclose the magnetic axis; precession from ∇B + curvature drifts. Compact Jacobi-elliptic closed forms exist (Brizard, arXiv:1011.2401), with P_φ = −(e/c)ψ + p_∥ B_φ/B conserved.
- **Drifts:** grad-B drift v_∇B = (μ/(qγ))(B×∇B)/B² = (m u_⊥²/(2qγB³))(B×∇B); curvature drift v_curv = (m u_∥²/(qγ))(R_c×B)/(R_c²B²); mirror ratio R_m = B_max/B_min, loss cone sinθ_lc = √(1/R_m).

### Part E — Runaway-electron physics layer
- **Dreicer field:** E_D = n_e e³ lnΛ/(4πε₀² k_B T_e) (equivalently E_D = E_c · m_ec²/(k_B T_e)).
- **Connor–Hastie critical field:** E_c = n_e e³ lnΛ/(4πε₀² m_e c²) (Connor & Hastie, Nucl. Fusion 15, 415, 1975). Above E_c all electrons with p > p_c run away, with p_c ≈ 1/√(E/E_c − 1) (in units of m_ec).
- **Relativistic collisional drag:** F_coll = e E_c (v/c)⁻² for superthermal electrons; critical speed v_c = c√(E_c/E). Single-particle test: a particle with E > E_c and p > p_c accelerates indefinitely; with E < E_c it decelerates — a clean validation of the drag term.
- **Avalanche (Rosenbluth & Putvinski, Nucl. Fusion 37, 1355, 1997):** (dn_RE/dt)_ava ≈ C(E,Z_eff) · n_RE/(2 lnΛ τ_c) · (E/E_c − 1); the growth rate is linear in E/E_c well above threshold. Here τ_c = 4πε₀² m_e² c³/(e⁴ n_e lnΛ).
- **Synchrotron radiation reaction (Landau–Lifshitz):** ignoring E and advective terms, F_R = (1/(γτ_r))[(p×b̂)×b̂ − (1/(m_ec)²)(p×b̂)² p], with τ_r = 6πε₀(m_ec)³/(e⁴B²). In azimuthally symmetric spherical momentum space: dp/dt = −(γp/τ_r)(1 − ξ²), dξ/dt = ξ(1 − ξ²)/(τ_r γ), ξ = p_∥/p (Beidler et al., arXiv:2007.15785, Eqs. 24–25).
- **Momentum-space attractor (fixed-point test):** with time in τ_c, momentum p in m_ec, E in E_c, and α ≡ σ_r = τ_c/τ_r, the deterministic (p,ξ) equations (Decker et al., Plasma Phys. Control. Fusion 58, 025016, 2016; Hirvijoki et al., J. Plasma Phys. 81, 475810502, 2015) are
  - dp/dt = ξE − C_F(p) − σ_r γ p (1 − ξ²), γ = √(1 + p²), with C_F → 1/v² at high energy;
  - dξ/dt = (√(1 − ξ²)/p)(−E + σ_r (p²/γ) ξ).
  The **O-point** (stable attractor) is the simultaneous null dp/dt = dξ/dt = 0. On axis (ξ = 1) the critical momentum is p_c = (E − 1)^{−1/2}; the asymptotic critical pitch is ξ_c = E⁻¹; the perpendicular runaway boundary is p_⊥0² = (E − 1)/σ_r; the bump (attractor) location scales as p_∥b ∝ [2/(1 + Z_eff)] · [(1 + σ_r)/σ_r] · (E − 1). This is an excellent physics validation with an analytic fixed point.
- **Two-threshold theory (Aleynikov & Breizman, Phys. Rev. Lett. 114, 155001, 2015):** the effective sustainment field (normalized to E_c) is E₀ ≈ 1 + [(Z+1)/√τ̄_rad] · √(1/8 + (Z+1)²/τ̄_rad)/6, with τ̄_rad = τ_rad/τ_c; E₀ > E_c always, and the avalanche onset E_a > E₀ produces hysteresis. Reviewed in Breizman, Aleynikov, Hollmann & Lehnen, Nucl. Fusion 59, 083001 (2019).
- **Collision-operator verification:** verify pitch-angle scattering by relaxation of an anisotropic beam to isotropy at the analytic ν_D rate; verify energy drag by relaxation to a Maxwell–Jüttner distribution; check the H-theorem/entropy increase; compare distribution moments to analytic Fokker–Planck relaxation rates. Ensemble statistical error ∝ 1/√N. SDE integration: Euler–Maruyama strong order 1/2 / weak order 1; Milstein strong order 1 (Dimits et al.; requires the area-integral terms). Note the order reduction relative to the deterministic pusher.

### Part F — Software / numerical hygiene
- **Non-dimensionalization:** normalize to c, Ω_c⁻¹, and r_L (or to τ_c and E_c for the RE physics layer). Test invariance under a change of normalization: the same physical trajectory must result.
- **Floating-point:** compute γ from u via γ = √(1 + |u|²/c²) (stable) rather than 1/√(1 − v²/c²) (catastrophic cancellation as v→c). For γ − 1 at both limits use the algebraically stabilized form γ − 1 = (|u|²/c²)/(1 + √(1 + |u|²/c²)). Store and advance u = γv (proper velocity), never v, to avoid precision loss at ultra-relativistic γ. PlasmaPy's RelativisticBorisIntegrator adopts exactly this proper-velocity formulation.
- **Method of Manufactured Solutions (MMS):** applicable — add an analytic forcing S(t) to the EoM so a chosen (u_MS(t), x_MS(t)) is an exact solution, then verify the observed order of accuracy is 2 (V&V methodology per Oberkampf & Roy; Salari & Knupp). For the field solver, MMS on the Cerfon–Freidberg Grad–Shafranov solution verifies field evaluation independently.
- **Property-based testing (hypothesis):** invariants — |v| conserved in pure B, rotation-matrix orthogonality RᵀR = I, γ unchanged by the magnetic step, energy monotonic under pure drag.
- **Regression / golden-file / CI:** pytest fixtures, parametrized Δt sweeps, numpy.testing.assert_allclose with explicit rtol/atol, seed control for stochastic tests, pytest-regressions for golden files, pytest-benchmark for performance regression.
- **Libraries:** numba/cython for pusher speed; h5py/xarray for output; pint or astropy.units for dimensional checks; PlasmaPy (BorisIntegrator / RelativisticBorisIntegrator plus the plasma formulary) as an independent unit-level cross-check; hypothesis for property testing. KORC (ORNL; Carbajal et al., Phys. Plasmas 24, 042512, 2017) is the reference RE full-orbit code for physics cross-verification.

## The Test Hierarchy (T0–T6)
For each test: purpose · failure mode caught · governing equations · analytic reference · error metric · expected convergence/tolerance · suggested parameters.

**T0 — Pure numerics / unit tests.** *Purpose:* catch index/sign/precision bugs.
- T0.1 Rotation orthogonality: RᵀR = I to 1e-14 (catches rotation-matrix sign errors).
- T0.2 γ-from-u round-trip and γ − 1 stability at v/c = 1e-8 and γ = 1e3 (catches cancellation).
- T0.3 Single-step vs hand-computed golden value.
- T0.4 Time-reversibility: N forward + N backward steps; ‖x_N − x₀‖ ≤ 1e-12 (catches half-step/kickback initialization bugs).
- T0.5 Volume preservation: finite-difference the 6×6 one-step Jacobian ∂(x,u)ⁿ⁺¹/∂(x,u)ⁿ; det = 1 to 1e-12.
- T0.6 Normalization invariance.

**T1 — Analytic single-particle, uniform fields.** *Reference:* Part C + matrix exponential.
- T1.1 Uniform B: check r_L, Ω_c; conserve |v|, γ, μ to machine precision; orbit closure; radius error vs Δt (slope 2 on log–log); gyrophase error ∝ (ω_cΔt)²; long-time energy drift over 10⁵–10⁷ gyroperiods (bounded, no secular growth). Suggested: B = 1 T, electron, v_⊥/c = 0.5.
- T1.2 Uniform E parallel acceleration to γ = 10²–10³ against the hyperbolic solution; L2 error slope 2.
- T1.3 E∥B helical acceleration; adiabatic-invariant checks.
- T1.4 **Crossed E×B — THE discriminator:** measure drift-velocity error vs γ and Δt; Boris error grows ~1 order of magnitude per increasing Δt while Vay/HC stay near machine precision. Metrics: L2 position, L∞, |Δγ/γ|, |Δp/p|. Suggested: E/cB = 0.5 and 0.999999 (drift-frame γ ≈ 707), lab γ up to 10⁵.
- Tolerances: relative energy error in pure B < 1e-12; convergence slope 2.0 ± 0.1.

**T2 — Analytic non-uniform fields.**
- T2.1 Grad-B drift vs v_∇B formula.
- T2.2 Curvature drift vs v_curv.
- T2.3 Magnetic mirror: bounce motion, μ conservation, loss cone; scan ρ/L to observe adiabaticity breakdown.
- T2.4 Field-interpolation order study: feed interpolated vs analytic field; verify the interpolation error does not degrade the pusher order below 2 (use ≥ 2nd-order interpolation).

**T3 — Tokamak geometry & invariants.** *Reference:* Solov'ev/Cerfon–Freidberg ψ.
- T3.1 Field evaluation vs analytic ψ to 1e-12.
- T3.2 **p_φ conservation (time-centred) over 10⁶ transits — primary diagnostic**; require bounded, not secular, drift.
- T3.3 Energy + μ conservation.
- T3.4 Passing vs trapped classification against v_∥/v < √(2ε/(1+ε)); banana width vs Δ_b ~ qρ_θ/√ε; bounce frequency ω_b; precession.
- T3.5 Reproduce a published KORC/ASCOT/ORBIT figure — e.g. KORC synchrotron "hot spots" at banana tips (del-Castillo-Negrete et al., Phys. Plasmas 25, 056104, 2018).

**T4 — Added physics terms.**
- T4.1 Drag: E > E_c runs away, E < E_c decelerates; verify p_c.
- T4.2 Synchrotron RR: relaxation of pitch toward the attractor; verify the dp/dt, dξ/dt forms.
- T4.3 Fixed-point/attractor test: locate the O-point (dp/dt = dξ/dt = 0) and compare to analytic scalings; verify the two-threshold E₀.
- T4.4 Bremsstrahlung drag if implemented.

**T5 — Statistical / ensemble (collisions).**
- T5.1 Pitch-angle isotropization rate vs analytic ν_D.
- T5.2 Relaxation to Maxwell–Jüttner; moments; H-theorem.
- T5.3 1/√N convergence in particle number.
- T5.4 SDE order: Euler–Maruyama strong 1/2, Milstein strong 1 (with area-integral terms); seed-controlled, fixed Brownian path.

**T6 — Full RE benchmarks vs published results.**
- T6.1 Dreicer growth rate vs E/E_D compared to Connor–Hastie and to kinetic codes CODE/LUKE/NORSE.
- T6.2 Avalanche growth rate vs Rosenbluth–Putvinski.
- T6.3 Synchrotron-limited RE energy vs Aleynikov–Breizman/Decker/Hirvijoki.
- T6.4 End-to-end comparison to KORC (Carbajal et al. 2017) and to momentum-attractor/synchrotron measurements (Paz-Soldan et al.; Stahl et al.).

## Recommendations
1. **Adopt Higuera–Cary as the production pusher** (volume- *and* E×B-preserving); keep Boris and Vay implemented behind a flag purely for the T1.4 discriminator regression. Add the Zenitani–Umeda gyrophase-corrected variant if gyrophase accuracy is important for your application (e.g. resolving cyclotron-scale synchrotron phase).
2. **Stage the CI:** run T0–T1 on every commit (seconds); T2–T3 nightly; T4–T6 on release. Gate merges on convergence-slope tests (2.0 ± 0.1) and on p_φ non-drift.
3. **Thresholds that change decisions:** (a) if the T1.4 Boris E×B error at γ = 100 exceeds ~1% (it will), that mandates HC/Vay for any relativistic crossed-field regime; (b) if T3.2 p_φ shows secular (linear-in-time) drift rather than bounded oscillation, the time-centring of p_φ or the field-interpolation order is wrong; (c) if T5.4 yields strong-order < 0.9 with Milstein, the area-integral terms are missing; (d) if T1.1 energy drifts secularly, the reference RK4 is behaving correctly but your Boris implementation has broken volume preservation (check the rotation).
4. **Cross-verify** at the unit level against PlasmaPy's Boris integrators and at the physics level against KORC; use scipy.integrate.solve_ivp (DOP853, rtol = 1e-12) and the matrix exponential as independent high-order references. Reproduce at least one published figure end-to-end before claiming validation.

## Caveats
- Boris is **not** symplectic (Qin et al. 2013) — claim only volume preservation and bounded energy error, never symplecticity. Higuera–Cary is likewise volume-preserving but not symplectic.
- The p_φ sign convention and the ψ definition (poloidal flux vs flux/2π; Gaussian −Zeψ/c vs SI +qψ) vary between references — fix yours explicitly and validate with a known passing orbit before trusting the diagnostic.
- The (p,ξ) attractor formulas from Decker/Hirvijoki are scalings/lower bounds, not exact equalities; the exact O-point must be obtained by solving the null conditions numerically. Aleynikov–Breizman instead use a pitch-averaged 1-D momentum continuity equation (not the two-ODE system), and their E₀ fit is quoted as valid for 1 < Z < 30, τ̄_rad > 5, to ~5% accuracy.
- MMS for particle ODEs is less standard than for PDEs but valid; add the manufactured forcing at the correct time-centring (half-step for the velocity equation) or you will spuriously measure order 1.
- Watch for the relativistic E×B "electric-dominated" regime (E²/c² > B²): there is **no** bounded drift — the particle accelerates indefinitely, so do not use a bounded-drift reference there.
- The Landau–Lifshitz radiation-reaction force is itself a perturbative reduction of Lorentz–Abraham–Dirac and can develop unphysical runaway solutions when the reaction force approaches the Lorentz force in the instantaneous rest frame; verify you are within its domain of validity for the B and γ used.