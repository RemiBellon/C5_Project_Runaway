# CLAUDE.md — Instructions pour Claude Code

Ce fichier définit **comment tu travailles sur ce projet**. Il prime sur tes habitudes par
défaut. Lis-le entièrement avant toute action.

---

## 0. Ce qu'est ce projet

`ppf_c5_runaways` est un code Python de suivi d'orbite complète (*full-orbit*) pour les **électrons
découplés (runaway electrons)** dans les plasmas de tokamak. Il implémente et compare les
intégrateurs relativistes Boris, Vay, Higuera–Cary et RK4, puis y ajoute les collisions et
la réaction de rayonnement synchrotron. Objectif final : un article scientifique.

Il est développé par un physicien des plasmas qui **apprend le développement Python
professionnel en même temps qu'il construit le code**. Ton rôle n'est donc pas de produire
du code vite, mais de produire **une seule brique à la fois, de qualité professionnelle,
que l'utilisateur comprend entièrement et peut relire ligne à ligne**.

**Tu réponds en français. Tu écris le code, les noms, les commentaires et les docstrings en
anglais** (le dépôt est destiné à la communauté internationale).

La bibliographie de référence est dans `biblio/` (Boris 1970, Vay 2008, Higuera & Cary
2017, Zenitani & Umeda 2018, Ripperda et al. 2018, Breizman et al. 2019, Chew 2023). Quand
un choix numérique vient d'un de ces articles, **cite-le dans la docstring** (auteur, année,
équation).

---

## 1. RÈGLE FONDAMENTALE — une seule unité à la fois

> **Tu écris UNE fonction OU UNE classe par échange. Puis tu t'arrêtes.**

Ce n'est pas une préférence stylistique, c'est la contrainte structurante du projet.

**Interdits absolus :**
- écrire plusieurs fonctions ou plusieurs classes dans une même réponse ;
- créer plusieurs fichiers dans une même réponse ;
- ajouter des méthodes « pendant que j'y suis » ;
- laisser des `TODO`, des `pass`, ou des stubs pour du code à venir ;
- anticiper un besoin futur (« je prépare déjà les collisions ») ;
- refactorer du code existant qui n'était pas l'objet de la demande.

**Exception unique** : méthodes triviales et indissociables de la classe demandée
(`__repr__`, `__eq__`, une propriété d'une ligne). Une méthode contenant de la logique
métier compte comme une unité à part entière.

Si la demande implique manifestement plusieurs unités, **tu ne les écris pas** : tu proposes
le découpage en liste ordonnée, tu expliques pourquoi cet ordre, et tu demandes par
laquelle commencer.

---

## 2. Protocole d'écriture — le contrat en 4 temps

Pour **chaque** unité, sans exception.

**Temps 1 — Annoncer** (5 lignes max, puis tu t'arrêtes et tu attends validation) :
- **Où** : chemin du fichier, et pourquoi ce module-là ;
- **Quoi** : signature exacte envisagée, avec les types et les formes de tableaux ;
- **Rôle physique/numérique** : ce que ça calcule, dans quelles unités SI ;
- **Interface** : ce qui l'alimente en amont, ce qui la consomme en aval ;
- **Test** : quel test la validera, et contre quelle **vérité indépendante du code**.

Si l'utilisateur corrige la signature, tu repars du temps 1.

**Temps 2 — Écrire le code.** Une seule unité, docstring numpydoc complète, unités SI
indiquées, formes de tableaux documentées.

**Temps 3 — Écrire le test.** Dans `tests/`, en miroir du module. Le test accompagne
toujours le code dans la même réponse : **une unité livrée sans test n'est pas livrée**.

**Temps 4 — Expliquer et signaler** : 3 à 6 lignes sur les choix non évidents (pas une
paraphrase du code), la commande exacte à lancer, les limites et hypothèses, puis la
proposition de l'unité suivante **sans l'écrire**.

---

## 3. Niveau de code attendu

Tu écris comme un développeur expérimenté en méthodes numériques **qui écrit pour être relu
par un physicien**. Les deux contraintes comptent autant.

| Attendu | Contre-exemple à éviter |
| --- | --- |
| Fonctions courtes, une responsabilité | une fonction qui charge, intègre et trace |
| Annotations de type sur toutes les signatures publiques | `def push(cfg):` |
| Docstring numpydoc avec unités SI et formes `(n_part, 3)` | `"""Push the particles."""` |
| Noms proches de la notation physique (`u`, `gamma`, `b_field`) | `tmp`, `data2`, `x1` |
| Validation des entrées avec message d'erreur utile | `assert n > 0` |
| Constantes physiques nommées, importées de `constants.py` | `1.602e-19` dans une formule |
| Vectorisation NumPy **sur les particules** | boucle Python sur les particules |
| Boucle Python lisible **sur le temps** | vectorisation acrobatique du pas de temps |

**Astuces interdites** : one-liners « malins », `lambda` complexes, métaclasses, décorateurs
maison, monkey-patching, héritage sur plus de deux niveaux, abstractions créées « au cas
où ». La sur-ingénierie est un défaut aussi grave que le code bâclé.

Si tu hésites entre deux formulations, **choisis celle qu'un physicien comprend en une
lecture**, même si elle est deux lignes plus longue.

---

## 4. Architecture — où va quoi

```
src/ppf_c5_runaways/
├── constants.py     constantes physiques SI nommées (c, e, m_e, eps_0, mu_0)
├── config.py        schéma pydantic du .yaml + chargement/écriture validés
├── fields.py        champs E(x,t), B(x,t) analytiques : uniforme, ExB, miroir, tokamak
├── pushers.py       boris / vay / higuera_cary / rk4 — même signature, un registre
├── forces.py        collisions et réaction de rayonnement (étape 4 uniquement)
├── integrate.py     boucle en temps pure : tableaux en entrée, tableaux en sortie
├── diagnostics.py   énergie, gamma, invariants, erreurs vs solution analytique
├── io.py            dossiers de résultats horodatés, écriture/lecture xarray→netCDF
├── run.py           orchestration : lit le .yaml, appelle integrate, écrit sur disque
└── plotting.py      figures, produites à partir d'un dossier de résultats
tests/               miroir exact de src/ppf_c5_runaways/
configs/             fichiers .yaml de simulation
main.py              poste de pilotage de l'utilisateur
```

**Règles de dépendance, non négociables :**
- `constants` et `config` ne dépendent de rien d'autre dans le projet.
- `fields` et `pushers` sont **purement numériques** : NumPy seul, aucune I/O, aucun objet
  de configuration. Ils prennent et rendent des `np.ndarray`.
- `integrate` ne connaît que `fields`, `pushers`, `forces`. **Il n'écrit jamais sur disque.**
  C'est ce qui le rend testable sans fichier.
- `run` est la **seule** couche qui orchestre et la **seule** qui écrit sur disque.
- `plotting` ne lit **que** des dossiers de résultats. Il n'importe jamais `integrate` ni
  `pushers` : on doit pouvoir refaire une figure sans relancer un calcul.

Si une unité demandée ne rentre dans aucun module, **signale-le avant d'écrire**.

**API publique attendue** (c'est l'ergonomie de l'utilisateur, respecte-la) :

```python
from ppf_c5_runaways import run, plotting

result_dir = run.pusher("configs/uniform_b.yaml")
plotting.gamma_vs_time(result_dir)
```

---

## 5. Conventions physiques et numériques — FIGÉES

Ne les change jamais sans le dire explicitement.

- **Unités SI partout.** Les conversions se font à la frontière (lecture du `.yaml`),
  jamais au milieu d'un calcul. Pas de `pint` ni d'unités typées dans les boucles.
- **Variable d'état : `u = gamma * v` (vitesse propre, m/s), jamais `v`.**
  `gamma = sqrt(1 + |u|^2/c^2)`. Motif : à 20 MeV, `v` est indiscernable de `c` en float64.
- **Charge signée portée explicitement** : `q = -e` pour l'électron. N'écris jamais `|q|`
  dans un pusher — le sens de giration en dépend.
- **Décalage leapfrog figé** : positions `x^n` aux pas entiers, vitesses `u^{n+1/2}` aux
  demi-pas. L'initialisation **exige un demi-pas arrière** sur `u`. Tout diagnostic à un pas
  entier utilise `u^n = (u^{n-1/2} + u^{n+1/2}) / 2` : ignorer ce recentrage détruit
  silencieusement l'ordre 2. Rappelle cette convention dans la docstring des fonctions
  concernées.
- **Formes de tableaux** : état `(n_part, 3)` en float64, C-order. Sorties temporelles
  `(n_saved, n_part, 3)`. Vectorisation sur l'axe particule, boucle Python sur le temps.
- **Interface champ** : `field(x, t) -> (E, B)` avec `x` de forme `(n_part, 3)` et
  `E`, `B` de forme `(n_part, 3)`. Aucun champ ne lit la configuration.
- **Interface pusher** : `push(x, u, field, dt, q, m) -> (x_new, u_new)`, un seul pas.
  Les quatre schémas partagent exactement cette signature.
- **Boris** : demi-accélération électrique, rotation magnétique, demi-accélération, avec
  `t = q*B*dt/(2*m*gamma_minus)` et `s = 2t/(1+|t|^2)` (Ripperda 2018, éq. 10–12). La
  variante à rotation exacte de Zenitani & Umeda (2018) est une **option à comparer**, pas
  un remplacement silencieux.
- **Vay (2008)** préserve la dérive E×B mais **pas le volume de l'espace des phases**.
  **Higuera–Cary (2017)** préserve les deux. Boris préserve le volume mais **pas** la
  dérive E×B. Ces trois faits sont les résultats que les tests doivent reproduire.
- **RK4** est le témoin non structurel : ordre 4 en temps, mais dérive séculaire de
  l'énergie. Il sert de contraste, pas de référence de production.
- **Aléatoire** (collisions) : `numpy.random.Generator` avec graine explicite, stockée dans
  les métadonnées du run. Jamais `np.random.seed` global.
- **Pas de temps fixe** à ce stade. Un pas adaptatif brise la symétrie temporelle du
  schéma : si on y vient, ce sera une décision explicite et documentée.
- **Aucun état global, aucun mutable au niveau module.**

---

## 6. Tests — ce qui compte comme une vérification

Un test qui vérifie seulement que le code ne plante pas n'a aucune valeur ici. Chaque test
confronte le résultat à une **vérité indépendante du code**. Par ordre de force de preuve :

1. **Propriété exacte du schéma discret** (attendue à la précision machine, ~1e-14) :
   conservation de `|u|` dans un B pur, réversibilité temporelle (N pas en avant, N pas avec
   `-dt`, retour à l'initial), préservation du volume via le déterminant jacobien du pas.
2. **Solution analytique** : giration uniforme (`r_L = gamma*m*v_perp/(|q|*B)`,
   `T_c = 2*pi*gamma*m/(|q|*B)`), accélération par E uniforme parallèle à B, dérive E×B,
   fréquence numérique de Boris `tan(omega_num*dt/2) = omega_c*dt/2`.
3. **Ordre de convergence** : raffinement en `dt`, pente observée 2 (Boris/Vay/HC) ou 4
   (RK4) à ±0.1. C'est le seul test qui détecte une erreur de formulation du schéma.
4. **Invariants physiques sur temps long** : moment magnétique `mu` dans un miroir
   (erreur **bornée**, pas séculaire), moment canonique toroidal `p_phi` en géométrie
   axisymétrique, énergie en champ magnétostatique.
5. **Référence publiée** : cas-tests de Ripperda et al. 2018 (E×B, champ *force-free*,
   miroir, dipôle, point nul), section de Poincaré de Higuera & Cary 2017 (champ cisaillé
   `E = a*x*ex`, `B = b*y*ex`, `a=1`, `b=2`, `H=4`), cas de Vay 2008.
6. **Intégrateur de référence indépendant** : `scipy.integrate.solve_ivp` (DOP853,
   `rtol=1e-12`) sur un champ arbitraire, comme vérité numérique de substitution.
7. **Property-based** (`hypothesis`) : pour des `B`, `u`, `dt` aléatoires, `|u|` invariant
   et `gamma >= 1`.
8. **Non-régression** (`pytest-regressions`) : valeur figée, en dernier recours seulement.

Les trois régimes doivent être couverts : non relativiste (`v/c ~ 1e-3`), modérément
relativiste (`gamma ~ 2`), ultra-relativiste (`gamma ~ 100`, soit ~50 MeV, régime runaway).

**Règles de rédaction des tests :**
- Un fichier de test par module, en miroir exact de `src/`.
- Les tolérances sont **explicites et justifiées en commentaire** : jamais
  `assert abs(a-b) < 1e-3` sans dire d'où vient `1e-3` (précision machine ? ordre du
  schéma × `dt^2` ? incertitude de la référence publiée ?).
- Marqueurs : `@pytest.mark.slow` (> quelques secondes), `@pytest.mark.convergence`,
  `@pytest.mark.physics`, `@pytest.mark.regression`.
- **Ne modifie jamais une tolérance pour faire passer un test.** Relâcher un seuil est une
  falsification. Si un test échoue : rapporte, diagnostique, propose.

Commandes : `uv run pytest -m "not slow"` (itération), `uv run pytest` (complet).

---

## 7. Outillage et environnement

- Gestionnaire : **uv**. Tout s'exécute via `uv run ...`. Ne propose jamais `pip install`,
  `python -m venv`, ni `conda`.
- Python 3.14, Linux x86-64 et macOS (Apple Silicon inclus).
- Avant de conclure une réponse contenant du code :
  `uv run ruff check --fix . && uv run ruff format . && uv run pytest -m "not slow"`.
- **N'ajoute aucune dépendance sans la demander explicitement**, en justifiant (apport,
  poids, maturité, licence) et en attendant l'accord. Le noyau doit rester petit.
- Ne modifie pas `pyproject.toml` de ta propre initiative.

---

## 8. Ce que tu ne fais jamais sans qu'on te le demande

- Lancer une simulation longue ou un scan paramétrique.
- Écrire dans `results/`.
- Créer ou modifier `main.py` — c'est le poste de pilotage de l'utilisateur.
- Committer, pousser, créer une branche ou un tag.
- Renommer, déplacer ou supprimer des fichiers existants.
- Réécrire une fonction que l'utilisateur a modifiée à la main.
- Générer de la documentation, des notebooks ou des exemples.

---

## 9. Honnêteté technique

Projet de recherche destiné à publication : une erreur silencieuse coûte des mois.

- Si une approche te semble fragile, **dis-le avant d'écrire**, pas après.
- Si tu n'es pas sûr d'une formule, d'une convention de signe ou d'un résultat publié,
  **dis-le explicitement**. « Je ne suis pas certain de ce point, à vérifier contre
  [référence] » est une réponse acceptable et utile. Du code faux qui tourne ne l'est pas.
- Ne prétends jamais avoir vérifié un résultat numérique que tu n'as pas exécuté.
- Signale les hypothèses implicites (champ statique, particule-test sans rétroaction,
  absence de collisions, non-uniformité négligée).

---

## 10. Étape courante

> **Étape 0 — Squelette du projet, schéma de configuration, I/O des résultats.**
>
> Objectif : charger un `.yaml` validé, créer un dossier de résultats horodaté contenant la
> configuration et les métadonnées du run.
>
> Critère de passage : round-trip de configuration (écriture → lecture → égalité) et
> création du dossier de résultats, tous deux couverts par des tests.

*(Section mise à jour par l'utilisateur. Elle définit le périmètre autorisé : ne propose
jamais de code appartenant à une étape ultérieure.)*

**Feuille de route** — aucune étape ne commence avant que la précédente soit vérifiée :

0. Squelette, config YAML, I/O résultats, constantes physiques.
1. Champs analytiques + les quatre pushers (Boris, Vay, Higuera–Cary, RK4), même interface.
2. Vérification systématique contre la littérature : invariants, préservation du volume,
   ordre de convergence, cas-tests de Ripperda 2018, trois régimes relativistes.
3. Configuration tokamak : géométrie axisymétrique, champ toroïdal + poloïdal, orbites
   passantes/piégées (bananes), conservation de `p_phi`, largeur d'orbite.
4. Nouveautés du projet : collisions (opérateur de Fokker–Planck test-particule) et
   réaction de rayonnement synchrotron ; comparaison rigoureuse à la littérature.
