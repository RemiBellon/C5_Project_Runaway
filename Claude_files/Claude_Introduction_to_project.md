# Runaway electrons dans les tokamaks : un projet Python "from scratch" avec Boris pusher — état de l'art, sujets candidats et niches publiables

## TL;DR
- **Recommandation n°1** : construire un traceur de particules test full-orbit relativiste (Boris → Higuera–Cary) en géométrie tokamak analytique (Solov'ev/Cerfon–Freidberg) et en faire une **étude systématique de la validité de l'approximation centre-guide (GC) pour les électrons runaway en fonction de l'énergie et de l'angle d'attaque** — c'est le sujet le plus faisable sur laptop, le mieux documenté, et clairement publiable (AJP/EJP/CPC ou Physics of Plasmas selon l'ambition).
- Le fil rouge naturel de montée en difficulté est : (1) gyration en champ uniforme + conservation d'énergie → (2) dérives (∇B, courbure, E×B) + invariance de μ → (3) orbites toroïdales complètes (passing/banana, moment canonique p_φ) → (4) accélération runaway + freinage synchrotron (force dissipative) → (5) collisions Monte-Carlo relativistes → (6) champ perturbé/stochastique (cartes de Poincaré, pertes).
- Deux **pièges numériques dominent** tout : la raideur ω_c·Δt ≪ 1 (la gyro-période relativiste est ≈10⁻¹¹ s alors que le temps d'accélération runaway est ≈1 s, soit >10¹¹ pas pour une trajectoire complète) qui impose de choisir intelligemment le régime étudié, et le fait que **la force de réaction de rayonnement casse la structure volume-preserving du Boris pusher**, ce qui ouvre une question méthodologique publiable en soi.

## Key Findings

1. **Le sujet le plus mûr pour un laptop est la comparaison full-orbit vs guiding-center.** La littérature a établi (Liu, Wang & Qin 2016) puis raffiné (C. Liu et al. 2018) l'existence d'un "collisionless / neoclassical pitch-angle scattering" qui invalide le modèle centre-guide standard pour les runaways très relativistes. Liu, Wang & Qin (2016, Nucl. Fusion 56, 064002) l'énoncent ainsi : *"It is discovered that the tokamak field geometry generates a toroidicity induced broadening of the pitch-angle distribution of runaway electrons. This collisionless pitch-angle scattering is much stronger than the collisional scattering and invalidates the gyro-center model for runaway electrons."* Leur simulation (champ type EAST : R₀=1,7 m, q=2, B₀=3 T, E_l=0,2 V/m) montre que la variation relative de |B| sur une gyro-période atteint 10 % après 0,4 s et dépasse 30 % après 1,7 s. Carbajal et al. (2017, KORC) quantifient indépendamment δ⟨B⟩ jusqu'à ≈15 % (et un changement de direction ⟨ΔB⟩ jusqu'à ≈60 %) sur une gyro-période dans un champ type DIII-D, un ordre de grandeur moindre pour ITER. Le débat n'est PAS clos de façon exhaustive : les seuils exacts en énergie/pitch/géométrie et le rôle de l'intégrateur restent un terrain d'étude quantitative accessible.

2. **Le Boris pusher est le bon point de départ, mais pas la fin de l'histoire.** Qin et al. (2013) ont prouvé qu'il conserve le volume de l'espace des phases (sans être symplectique), d'où l'absence de dérive séculaire en énergie et une borne globale sur l'erreur d'énergie. Pour le relativiste, Higuera–Cary (2017) est volume-preserving ET restitue correctement la dérive E×B (contrairement à Vay 2008, qui préserve E×B mais pas le volume). Comparer ces intégrateurs sur des trajectoires longues de runaways est un angle méthodologique légitime.

3. **La force de réaction de rayonnement est le point dur ET l'opportunité.** L'ajout du freinage synchrotron (forme Landau–Lifshitz, ou modèle de traînée −P_R v/v²) rend l'équation dissipative et détruit la conservation du volume de phase. Les schémas de splitting (Boris pour Lorentz + sous-pas pour la dissipation) sont l'approche standard mais leurs propriétés de conservation à long terme sont mal caractérisées — question ouverte.

4. **La physique runaway pertinente pour du test-particle léger est bien bornée** : mécanisme de Dreicer, avalanche de Rosenbluth–Putvinski, champ critique de Connor–Hastie, écrantage partiel des impuretés (Hesslow et al. 2018), limite d'énergie synchrotron, structure de l'espace des impulsions (attracteurs/anneaux sous rayonnement), transport en champ stochastique.

5. **Des ressources pédagogiques existent** (dépôts GitHub Boris, notes de cours guiding-center, tutoriels) mais aucune ne combine "from scratch + runaways + géométrie tokamak + étude de convergence d'intégrateur", ce qui laisse de la place pour un article pédagogique original.

## Details

### A. État de l'art — physique des runaways pour de la simulation test-particle légère

**Génération primaire (Dreicer).** Un champ électrique parallèle accélère les électrons dont la friction collisionnelle décroît en 1/v² ; au-dessus d'une vitesse critique ils "s'échappent". Réf. fondatrice : H. Dreicer, "Electron and Ion Runaway in a Fully Ionized Gas. I / II", Phys. Rev. 115, 238 (1959) et 117, 329 (1960).

**Limite relativiste et champ critique.** J. W. Connor & R. J. Hastie, "Relativistic limitations on runaway electrons", Nucl. Fusion 15, 415 (1975), DOI 10.1088/0029-5515/15/3/007 — introduit le champ critique E_c en dessous duquel aucun runaway n'est généré.

**Avalanche (génération secondaire).** M. N. Rosenbluth & S. V. Putvinski, "Theory for avalanche of runaway electrons in tokamaks", Nucl. Fusion 37, 1355 (1997), DOI 10.1088/0029-5515/37/10/I03 — collisions "knock-on" à grand angle multipliant exponentiellement la population ; source δ(ξ−ξ₁) souvent utilisée. Ce modèle sous-estime le taux d'avalanche près du seuil, en particulier quand l'énergie moyenne des runaways est modeste (voir Stahl et al. ; Embréus et al. ; et le "Runaway Electron Avalanche Surrogate", arXiv 2504.03201).

**Écrantage partiel des impuretés.** L. Hesslow et al., "Effect of partially ionized impurities and radiation on the effective critical electric field for runaway generation", Plasma Phys. Control. Fusion 60, 074010 (2018), DOI 10.1088/1361-6587/aac33e ; et "Generalized collision operator for fast electrons interacting with partially ionized impurities", J. Plasma Phys. 84, 905840605 (2018). Le champ critique effectif est drastiquement supérieur au Connor–Hastie classique, et dépasse même la valeur obtenue en remplaçant la densité d'électrons libres par la densité totale (libres + liés). Pertinent pour l'injection massive de gaz / shattered pellet injection (SPI). Un script Matlab de référence existe (github hesslow/Eceff).

**Rayonnement et espace des impulsions.** Le freinage synchrotron crée une limite d'énergie et des structures non-monotones : attracteurs de phase, distributions en anneau/"bumps" sous rayonnement dominant (voir Decker/Hoppe/Embréus ; Aleynikov & Breizman ; "Ring momentum distributions...", Phys. Plasmas 31, 052112 (2024)). Diagnostic expérimental : émission synchrotron.

**Orbites et transport.** La dérive fait que les orbites circulantes de runaways dérivent vers l'extérieur (par conservation du moment canonique toroïdal p_φ ; Guan, Qin & Fisch, "Phase-space dynamics of runaway electrons in tokamaks", Phys. Plasmas 17, 2010). Le transport en champ magnétique stochastique / RMP est un mécanisme de perte majeur (Carbajal et al. 2020 ; MARS-F pour ITER, avec ≈40 % de pertes en modèle de champ vacuum mais <5 % avec réponse plasma).

**Instabilités cinétiques.** Ondes whistler excitées par les runaways : D. A. Spong et al., "First Direct Observation of Runaway-Electron-Driven Whistler Waves in Tokamaks", Phys. Rev. Lett. 120, 155002 (2018) ; C. Liu et al., "Role of Kinetic Instability in Runaway-Electron Avalanches and Elevated Critical Electric Fields", Phys. Rev. Lett. 120, 265001 (2018). (Hors de portée d'un test-particle simple, mais bon contexte.)

**Revue clé à exploiter.** B. N. Breizman, P. Aleynikov, E. M. Hollmann & M. Lehnen, "Physics of runaway electrons in tokamaks", Nucl. Fusion 59, 083001 (2019), DOI 10.1088/1741-4326/ab1822 — LA revue de référence (particle orbits, guiding-center, resonant MHD perturbations, opérateurs de collision, stopping power, elastic scattering, réaction de rayonnement synchrotron, bremsstrahlung, pair production, thermal quench, transport, ITER). À lire en premier.

**Codes de référence (à NE PAS réimplémenter, à citer comme comparaison)** : KORC (ORNL, full-orbit + GC), DREAM (Hoppe, Embréus & Fülöp, "DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations", Comput. Phys. Commun. 268, 108098 (2021), DOI 10.1016/j.cpc.2021.108098), CODE, LUKE, APT (Accurate Particle Tracer), NORSE.

### B. Aspects numériques et méthodologiques

**Boris pusher.** Standard depuis 50 ans en PIC ; second ordre, une évaluation de champ par pas. Qin, Zhang, Xiao, Liu, Sun & Tang, "Why is Boris algorithm so good?", Phys. Plasmas 20, 084503 (2013), DOI 10.1063/1.4818428 : conserve le volume de phase (bien que non symplectique — les auteurs le prouvent explicitement), borne globale sur l'erreur d'énergie, mais l'erreur de phase (fréquence) croît linéairement dans le temps. Piège classique : erreur importante pour grand ω_c·Δt (S. E. Parker & C. K. Birdsall, "Numerical error in electron orbits with large ω_ce Δt", J. Comput. Phys. 97, 91 (1991)).

**Variantes relativistes.** J.-L. Vay, "Simulation of beams or plasmas crossing at relativistic velocity", Phys. Plasmas 15, 056701 (2008) : correcte pour E×B mais non volume-preserving (erreur possible sur le gyro-rayon). A. V. Higuera & J. R. Cary, "Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields", Phys. Plasmas 24, 052104 (2017), DOI 10.1063/1.4979989 : volume-preserving ET E×B correcte, coût similaire à Vay, ne diffère du Boris relativiste que par le calcul de γ — **recommandé pour les runaways**. Comparaison de référence très pédagogique : B. Ripperda et al., "A Comprehensive Comparison of Relativistic Particle Integrators", ApJS 235, 21 (2018) (Boris, Vay, Higuera–Cary, implicite, GC relativiste). Algorithmes volume-preserving d'ordre supérieur : He et al. (J. Comput. Phys. 281, 135, 2015 ; Phys. Plasmas 23, 092109, 2016), Zhang et al. (Phys. Plasmas 22, 044501, 2015, "secular relativistic dynamics").

**Force dissipative dans un schéma Boris.** L'ajout de la réaction de rayonnement (forme Landau–Lifshitz ; l'ALD/Lorentz-Abraham-Dirac a des solutions "runaway" non physiques) casse la structure volume-preserving. Approches : splitting (Boris pour Lorentz + sous-pas dissipatif), Boris modifié, solutions analytiques de la LL réduite (Heintzmann & Schrüfer 1973 ; Pétri 2021 pour les magnétosphères ; Li et al. 2021). Les schémas implicites symétriques dans le temps semblent mieux conserver les lois sur des runs longs (Elkina et al. 2014). C'est un vrai sujet ouvert : les propriétés de conservation à long terme des schémas dissipatifs pour runaways sont mal caractérisées.

**Full-orbit vs centre-guide (le point central).** 
- Liu, Wang & Qin, "Collisionless pitch-angle scattering of runaway electrons", Nucl. Fusion 56, 064002 (2016), DOI 10.1088/0029-5515/56/6/064002 (arXiv:1510.00780) : découverte par full-orbit que la géométrie toroïdale élargit la distribution en pitch, invalidant le modèle gyro-centre standard.
- C. Liu, H. Qin, E. Hirvijoki, Y. Wang & J. Liu, "Conservative magnetic moment of runaway electrons and collisionless pitch-angle scattering", Nucl. Fusion 58, 106018 (2018), DOI 10.1088/1741-4326/aad2a5 (arXiv:1804.01971) : *"we derive a new expression for the magnetic moment of relativistic runaway electrons, which is conserved significantly better than the standard one. The new result includes one of the second-order corrections in the standard guiding-center theory which, in case of runaway electrons with p∥≫p⊥, can peculiarly be of the same order as the lowest-order term."* Ce μ corrigé est bien conservé jusqu'à ≈80 MeV (au-delà, une simulation full-orbit redevient nécessaire), réconciliant l'effet avec l'ordering GC.
- Carbajal, del-Castillo-Negrete, Spong, Seal & Baylor, "Space dependent, full orbit effects on runaway electron dynamics in tokamak plasmas", Phys. Plasmas 24, 042512 (2017), DOI 10.1063/1.4981209 : article fondateur de KORC. *"full orbit calculations are used to explore the limitations of gyro-averaging in the relativistic regime. To explore the practical impact of the results, DIII-D and ITER-like parameters are used in the simulations."* Échelles temporelles couvertes : *"from the fast gyro-motion, ∼10⁻¹¹ s, to the observational time scales, ∼10⁻²→1 s."* Ils soulignent que le grand déplacement parallèle ρ∥ = v∥τ_e (≈0,6 m) peut être comparable à l'échelle de gradient L_B et signaler la faillite du gyro-averaging même si ρ_L est petit.
- Vérification récente : Y. Wang et al., "Calculation of collisionless pitch-angle scattering of runaway electrons with synchrotron radiation via high-order guiding-centre equation", J. Plasma Phys. (2022) — *"The magnetic moment µ is decided not only by the perpendicular momentum, but also by the parallel momentum and the magnetic field line curvature vector κ, thus providing the aforementioned collisionless momentum transfer."*

**Collisions Monte-Carlo relativistes.** Opérateur pitch-angle de A. H. Boozer & G. Kuo-Petravic, "Monte Carlo evaluation of transport coefficients", Phys. Fluids 24, 851 (1981) ; schémas de composition d'ordre 2 supérieurs au Boozer–Kuo-Petravic (Comput. Phys. Commun. 2013) ; opérateurs relativistes basés sur l'intégrale de collision Beliaev-Budker avec pas adaptatif (arXiv OSTI 1441004) ; modèle de collisions RE de Helander/Eriksson/Andersson (Phys. Plasmas 7, 4106, 2000), utilisé dans KORC. Approche test-particle : équation de Langevin / Euler–Maruyama pour la diffusion en pitch, avec une variante énergie-conservante (PubMed 33075918).

**Géométrie de champ analytique légère.** (1) Champ "circular concentric flux surfaces" (B_φ = B₀R₀/R, B_θ via un profil q(r)) — trivial à coder, suffisant pour orbites de base + ripple ; c'est exactement le modèle utilisé dans les études full-orbit runaway (arXiv 1605.08141 : R₀=1,7 m, a=0,4 m, q=2, B₀=2 T, E_l=0,2 V/m, Δt≈1,9×10⁻¹² s ≈ 1 % de T_c). (2) Équilibre de Solov'ev / A. J. Cerfon & J. P. Freidberg, "'One size fits all' analytic solutions to the Grad–Shafranov equation", Phys. Plasmas 17, 032502 (2010), DOI 10.1063/1.3328818 — forme fermée, aspect ratio/élongation/triangularité arbitraires, X-points possibles, idéal pour formes D. Ajout de ripple toroïdal : B_φ → B_φ(1+δ cos(N φ)). Perturbations résonnantes : superposition de modes (m,n) pour cartes de Poincaré (attention à ∇·B=0). Notes pédagogiques : Youjun Hu, "Guiding center motion in tokamaks" (benchmarks Solov'ev).

**Normalisation et vérification.** Normaliser le temps par ω_c⁻¹ ou la période toroïdale, les longueurs par R₀, l'impulsion par m_e c. Tests : conservation d'énergie en champ statique (Boris) ; invariance de μ = p_⊥²/(2m B) sur les orbites lentes ; conservation du moment canonique toroïdal p_φ = m R v_φ − eψ ; méthode des solutions manufacturées ; dérive E×B analytique.

### C. Ordres de grandeur du coût numérique (laptop)

- Gyro-période électronique relativiste : T_c = 2πγm_e/(eB) ≈ 2×10⁻¹¹ s (B≈2 T, γ modéré). Pas de temps full-orbit : Δt ≈ T_c/50–100 ≈ 2–4×10⁻¹³ s.
- Une trajectoire runaway "complète" (accélération sur ≈1 s) = ≈10¹¹–10¹² pas → **HORS de portée d'un laptop en full-orbit**. C'est pourquoi les plus grandes campagnes tournaient sur supercalculateur : Liu, Qin, Wang et al., "Largest Particle Simulations Downgrade the Runaway Electron Risk for ITER" (arXiv:1611.02362) ont fait *"simulations of 10⁷ sampled REs in 6D phase space, which involves simulation scale of 10¹⁸ particle-steps, the largest ever achieved in fusion research"* sur le *"Sunway TaihuLight supercomputer... equipped with more than 10 million cores... over 100 Pflops"*, aboutissant à une énergie moyenne maximale *"in the range of 150 MeV, less than half of previously predicted"* (le ripple réaliste améliore fortement le confinement : loss rate 2,6 % en config idéale à t=0,4 s contre ≈0 en config réaliste). **Conclusion pratique : ne PAS simuler la vie entière d'un runaway en full-orbit sur laptop.**
- Ce qui EST faisable sur laptop :
  - Trajectoires full-orbit sur 10⁴–10⁶ tours poloïdaux (10⁶–10⁸ pas) pour une particule → secondes à minutes en NumPy vectorisé/Numba.
  - Ensembles de 10³–10⁵ particules en GC (pas de temps ≈ période de bounce, 10³–10⁵ × plus grand qu'en full-orbit) pour statistiques de pertes → minutes à heures.
  - Cartes de Poincaré : 10²–10³ conditions initiales × 10³–10⁴ intersections → trivial.
- Stack recommandé : NumPy vectorisé sur l'ensemble de particules ; Numba (@njit) pour les boucles temporelles ; JAX si l'on veut du GPU/autodiff pour la vérification.

### D. Ressources pédagogiques existantes

- Dépôts GitHub éducatifs Boris : `gbogopolsky/boris-method` (Boris–Bunemann en Python, cas test E×B + configuration magnétique du tokamak Tore Supra à Cadarache) ; `particleincell.com` (tutoriel "Particle Push in Magnetic Field (Boris Method)", avec discussion du cas relativiste) ; `ORNL-Fusion/KORC` et `chalmersplasmatheory/DREAM` (codes complets, référence).
- Notes de cours guiding-center : Youjun Hu, "Guiding center motion in tokamaks" (notes + benchmarks Solov'ev, classification d'orbites, fréquence de bounce) ; notes Chalmers "Simulation of Charged Particle Orbits in Fusion Plasmas" (orbites banana/passing, dérives).
- Articles méthodologiques Boris : Wang, Zhang, Ye, Xu, "An improved Boris algorithm for charge particle orbit in tokamak plasmas" (IOP, 2025) ; "Phase Stability Analysis of Volume-preserving Algorithms for Accurate Single Particle Orbit Simulations in Tokamak Plasmas" (arXiv 2503.18482) ; "An analytic Boris pusher for plasma simulation" (Comput. Phys. Commun., 2022).
- Comparaison Boris vs GCA pédagogique (astro) : arXiv 2311.16182 (magnétosphères d'étoiles à neutrons).
- Livres : Birdsall & Langdon, *Plasma Physics via Computer Simulation* ; Hockney & Eastwood, *Computer Simulation Using Particles* ; Wesson, *Tokamaks* ; Freidberg, *Ideal MHD* et *Plasma Physics and Fusion Energy* ; Boozer (revues) ; Helander & Sigmar, *Collisional Transport in Magnetized Plasmas* (opérateurs de collision, indispensable pour la partie Monte-Carlo).

### E. Évaluation critique de 7 sujets de projet candidats

**Sujet 1 — Validité de l'approximation centre-guide pour runaways vs énergie (RECOMMANDÉ).**
- *Question* : à partir de quelle énergie/pitch/géométrie le μ standard cesse-t-il d'être adiabatique, et que gagne le μ corrigé de C. Liu 2018 ? Reproduire et cartographier l'effet de "collisionless pitch-angle scattering".
- *Montée* : Boris uniforme → dérives → orbites toroïdales → accélération + mesure de ΔB/B et de μ sur gyro-période → comparaison full-orbit / GC standard / GC corrigé.
- *Coût* : une à quelques particules, 10⁶–10⁸ pas ; parfaitement laptop.
- *Réfs* : Liu/Wang/Qin 2016, C. Liu et al. 2018, Carbajal et al. 2017, Wang et al. 2022, Breizman et al. 2019.
- *Publiable* : oui — une étude de convergence propre et une cartographie systématique (énergie × pitch × R₀/a × q, machine DIII-D vs ITER vs EAST) manque dans la littérature accessible ; angle AJP/EJP (pédagogique) ou Phys. Plasmas (si extension).
- *Pièges* : ω_c·Δt ≪ 1 ; définir μ proprement (standard vs corrigé avec κ) ; ne pas confondre erreur numérique et effet physique.

**Sujet 2 — Comparaison systématique d'intégrateurs pour trajectoires longues de runaways.**
- *Question* : Boris vs Vay vs Higuera–Cary vs RK4 vs volume-preserving d'ordre supérieur : erreur d'énergie/phase, conservation de p_φ et μ sur 10⁶⁺ tours, avec et sans E accélérateur.
- *Coût* : faible ; idéal pour du benchmark.
- *Réfs* : Qin et al. 2013, Higuera & Cary 2017, Vay 2008, Ripperda et al. 2018, He et al. 2015/2016.
- *Publiable* : CPC ou AJP/EJP comme article méthodologique/pédagogique ; l'originalité vient de l'application spécifique aux runaways (régime v_∥→c, suivi de p_φ) plutôt qu'aux cas génériques déjà publiés.
- *Pièges* : dérive de phase linéaire du Boris ; comparaison équitable à coût CPU égal.

**Sujet 3 — Intégration de la réaction de rayonnement dans un schéma Boris : propriétés de conservation (RECOMMANDÉ, plus ambitieux).**
- *Question* : comment le splitting Lorentz/dissipation affecte-t-il la structure de l'espace des phases ? Comparer splitting vs Boris modifié vs LL analytique sur la limite d'énergie synchrotron et les attracteurs de phase.
- *Montée* : Boris → +traînée synchrotron −P_R v/v² → mesure de la limite d'énergie → carte de l'espace des impulsions (point d'équilibre stable) → comparaison de schémas.
- *Coût* : faible à modéré.
- *Réfs* : Guan/Qin/Fisch 2010 (limite d'énergie, point fixe), Aleynikov & Breizman, Landau–Lifshitz, Wang/Qin/Liu 2016 (APT, limite synchrotron), Elkina et al. 2014 (schémas implicites LL).
- *Publiable* : oui — la caractérisation des propriétés de conservation des schémas dissipatifs pour runaways est un trou identifié.
- *Pièges* : la dissipation brise le volume-preserving ; choisir la forme de P_R cohérente (full-orbit vs GC).

**Sujet 4 — Statistiques de pertes d'orbite en présence de ripple toroïdal.**
- *Question* : fraction de runaways perdus vs amplitude de ripple δ, énergie, pitch ; comparaison full-orbit / GC.
- *Coût* : ensembles 10³–10⁵ particules en GC ; modéré.
- *Réfs* : Wang/Qin/Liu 2016 (ripple abaisse la limite d'énergie), "Largest Particle Simulations..." (arXiv 1611.02362, ITER : 2,6 % de pertes en config idéale, ≈0 en config réaliste), Carbajal et al. 2020 (stochastique).
- *Publiable* : possible si angle nouveau (ex. dépendance fine en δ à énergie modérée, ou effet full-orbit vs GC).
- *Pièges* : coût full-orbit prohibitif pour de vraies statistiques → rester en GC ou sur des temps courts.

**Sujet 5 — Cartes de Poincaré de runaways relativistes en champ perturbé.**
- *Question* : topologie (îlots, stochasticité, seuil de recouvrement de Chirikov) vue par des runaways de différentes énergies ; effet de la largeur d'orbite finie (finite-orbit-width vs field-line).
- *Coût* : faible (Poincaré = peu de particules, longues intégrations).
- *Réfs* : Carbajal et al. 2020 (Phys. Plasmas 27, 032502), Rechester & Rosenbluth 1978 (diffusion), del-Castillo-Negrete.
- *Publiable* : oui comme étude pédagogique/illustrative ; l'effet "orbite à largeur finie vs ligne de champ" est visuellement fort.
- *Pièges* : construire un champ perturbé à divergence nulle (∇·B=0) proprement.

**Sujet 6 — Effet du rayonnement synchrotron sur la topologie de l'espace des phases.**
- *Question* : reproduire les attracteurs/points fixes et la limite d'énergie en test-particle, puis explorer leur sensibilité aux paramètres (B, V_loop, Z_eff).
- *Coût* : faible.
- *Réfs* : Guan/Qin/Fisch 2010, Martín-Solís (momentum-space structure), Decker/Hoppe (bumps), "Ring momentum distributions..." Phys. Plasmas 31, 052112 (2024).
- *Publiable* : angle pédagogique clair (systèmes dynamiques dissipatifs) ; original si couplé à une carte 2D (p_∥, p_⊥) soignée.
- *Pièges* : distinguer traînée collisionnelle et traînée radiative.

**Sujet 7 — Transport de runaways en champ magnétique stochastique (diffusion).**
- *Question* : mesurer le coefficient de diffusion radiale D vs amplitude de perturbation et énergie ; tester le régime de Rechester–Rosenbluth.
- *Coût* : ensembles GC, modéré à élevé.
- *Réfs* : Carbajal et al. 2020, Rechester & Rosenbluth 1978, Särkimäki (orbit-following), "Confinement of passing and trapped runaway electrons..." (Nucl. Fusion, IOP ac75fd).
- *Publiable* : plus difficile (terrain occupé par KORC/ASCOT) — viser un angle pédagogique/méthodologique.
- *Pièges* : convergence statistique ; construction du champ stochastique.

### F. Recommandation d'architecture logicielle

Un unique module Python : (i) `fields.py` (champ uniforme, circular-concentric, Solov'ev/Cerfon–Freidberg, +ripple, +perturbations MHD) ; (ii) `pushers.py` (Boris, Boris relativiste, Vay, Higuera–Cary, RK4 de référence, GC relativiste) ; (iii) `collisions.py` (pitch-angle Monte-Carlo Boozer–Kuo-Petravic, slowing-down) ; (iv) `radiation.py` (traînée synchrotron, splitting) ; (v) `diagnostics.py` (énergie, μ, p_φ, Poincaré, pertes). NumPy vectorisé + Numba. Tests unitaires = solutions analytiques (gyration, E×B, μ).

## Recommendations

**Étape 0 (semaine 1) — "hello world".** Boris non relativiste, champ B uniforme. Vérifier gyro-rayon, gyro-fréquence, conservation d'énergie machine-precision sur 10⁶ tours. Ajouter E constant → dérive E×B analytique. Livrable : figure de conservation d'énergie Boris vs RK4 (RK4 dérive, Boris borné). *Seuil de passage : erreur d'énergie Boris bornée < 10⁻¹⁰ relatif sur 10⁶ tours.*

**Étape 1 (semaines 2–3) — dérives et invariants.** Champ non uniforme (∇B, courbure) ; vérifier dérives et invariance de μ. Passer au relativiste (Higuera–Cary). *Seuil : μ conservé à mieux que 1 % sur les orbites lentes ; dérive E×B numérique = analytique.*

**Étape 2 (semaines 4–6) — géométrie tokamak.** Champ circular-concentric puis Solov'ev/Cerfon–Freidberg. Orbites passing/banana, conservation de p_φ, classification d'orbites. *Seuil : p_φ conservé, orbites banana correctes, largeur d'orbite ∝ énergie.*

**Étape 3 (semaines 7–10) — cœur runaway.** Ajouter E_∥ accélérateur + traînée synchrotron. Mesurer la limite d'énergie et le point fixe dans (p_∥, p_⊥). **Basculer ici sur le Sujet 1 ou 3.** Mesurer ΔB/B et μ (standard vs corrigé) sur gyro-période en fonction de l'énergie → cœur de l'article.

**Étape 4 (semaines 11–14) — angle publiable.** Selon le sujet : cartographie GC-vs-full-orbit (Sujet 1), benchmark d'intégrateurs (Sujet 2), ou conservation sous dissipation (Sujet 3). Ajouter les collisions Monte-Carlo si le temps le permet.

**Étape 5 (semaines 15+) — rédaction.** Cibler EJP/AJP (angle pédagogique + méthode) ou Physics of Plasmas / preprint arXiv (si résultat physique neuf). Toujours situer par rapport à KORC/DREAM/APT et fournir le code (reproductibilité).

**Priorisation argumentée :**
1. **Sujet 1** (GC vs full-orbit) — meilleur rapport faisabilité/originalité/pédagogie. Un seul particule suffit, physique riche, débat vivant, forte base bibliographique.
2. **Sujet 3** (dissipation + Boris) — plus original méthodologiquement, un peu plus risqué.
3. **Sujet 2** (benchmark d'intégrateurs) — filet de sécurité toujours publiable en revue pédagogique, peut être fusionné avec le Sujet 1.

**Seuils qui changeraient la recommandation :** si vous disposez d'un GPU (JAX) et visez des statistiques d'ensemble → Sujets 4/7 deviennent viables. Si l'objectif prime "physique neuve" sur "pédagogie" → Sujet 3. Si le temps est très limité → Sujet 2 seul.

## Caveats
- **Le coût full-orbit interdit de simuler la vie complète d'un runaway sur laptop** ; il faut soit se limiter à des fenêtres temporelles courtes / peu de tours, soit passer en centre-guide pour les statistiques. C'est une contrainte structurelle (10¹¹–10¹² pas pour 1 s), pas un détail.
- **Le débat GC-breakdown est partiellement réconcilié** : C. Liu et al. (2018) montrent qu'un μ corrigé (incluant p_∥ et la courbure κ) est conservé jusqu'à ≈80 MeV. L'originalité doit donc porter sur une cartographie quantitative systématique ou sur l'effet de l'intégrateur, pas sur "découvrir" l'effet.
- Certaines valeurs (δB≈15 % DIII-D d'après Carbajal et al. 2017 ; seuil ≈80 MeV d'après C. Liu et al. 2018 ; ΔB 10 %/30 % après 0,4 s/1,7 s en champ EAST d'après Liu/Wang/Qin 2016) proviennent de configurations spécifiques et ne se transposent pas directement à d'autres machines ; vérifier chaque chiffre sur la source primaire avant de le reprendre dans l'article.
- Plusieurs mécanismes cités (whistler, avalanche auto-cohérente, MHD) exigent un modèle auto-cohérent (PIC/quasilinéaire) hors de portée d'un test-particle simple ; les inclure comme contexte, pas comme cible.
- Vérifier le numéro d'article/volume exact de Wang et al. 2022 (J. Plasma Phys., vérification de la théorie GC d'ordre élevé avec rayonnement) sur la page de la revue avant citation finale.
- Les revues AJP/EJP attendent une contribution pédagogique claire ; Physics of Plasmas/CPC attendent de la nouveauté physique ou méthodologique. Calibrer l'ambition en conséquence.