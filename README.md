# MBioChain : prototype du papier

Ce prototype génère une clé ECC (secp256k1) à partir de l'**iris**, de l'**empreinte digitale** et d'un **code PIN**, puis gère l'identité sur une blockchain (enregistrement, authentification, rotation de clé, récupération par des gardiens).

## Fichiers

| Fichier | Rôle |
|---|---|
| `mbiochain/ecc_codes.py` | Codes correcteurs : Hadamard RM(1,6) et Reed-Solomon sur GF(2^7), écrits sans bibliothèque externe |
| `mbiochain/bio_key.py` | Schéma proposé : fusion, permutation dépendant du PIN, *fuzzy commitment*, scrypt, HKDF, clé secp256k1 |
| `mbiochain/ledger.py` | Blockchain PoA simulée et contrat d'identité (REGISTER / ROTATE / RECOVER, défi-réponse) |
| `mbiochain/data.py` | Données synthétiques (test du code uniquement) et chargeur de vos vrais gabarits |
| `contracts/IdentityRegistry.sol` | Version Solidity du contrat, pour Ethereum (Sepolia, Ganache, Hardhat) si vous voulez mesurer le gas |
| `demo.py` | Scénario complet de bout en bout |
| `experiments.py` | Toutes les expériences du papier. Produit les CSV, les figures et `tables.tex` |

Dépendances : Python ≥ 3.10, `numpy`, `cryptography`, `matplotlib`.

```bash
pip install numpy cryptography matplotlib
python3 demo.py                      # scénario complet
python3 -m mbiochain.ecc_codes       # auto-test des codes correcteurs
```

## IMPORTANT : les données synthétiques ne sont pas des résultats

`python3 experiments.py --synthetic` sert **uniquement à vérifier que le code tourne**. Les FRR et FAR obtenus sur des gabarits synthétiques **ne doivent pas apparaître dans le papier**. Les tableaux II, III et IV du papier doivent être remplis avec vos tests sur de vraies bases.

## Faire les tests sur de vraies données

### 1. Bases conseillées (base multimodale « virtuelle »)

On associe le sujet *i* de la base iris au sujet *i* de la base empreintes. C'est une pratique courante, utilisée par exemple par Nagar et al. (2012).

- **Iris** : CASIA-IrisV4 (sous-ensemble Interval ou Lamp), ou IITD Iris. La base se demande en ligne à l'institut.
- **Empreintes** : FVC2002 DB1/DB2 ou FVC2004 DB1 (8 échantillons par doigt).

### 2. Extraire les gabarits binaires

- **Iris → IrisCode 8 × 256 = 2048 bits** : utilisez un outil open source, par exemple **USIT** (University of Salzburg Iris Toolkit) ou **OSIRIS**. Si l'outil produit un code de taille différente, ré-échantillonnez-le à 8 × 256.
- **Empreinte → vecteur binaire de 1024 bits** : binarisez une représentation de longueur fixe des minuties, par exemple une représentation spectrale des minuties ou un MCC (Minutia Cylinder-Code) binarisé et réduit à 1024 bits. Les minuties s'extraient par exemple avec **NBIS/MINDTCT** du NIST. Décrivez précisément votre méthode dans la section V-A du papier.

### 3. Organiser les fichiers

```
templates/
  iris/
    001/  s1.npy  s2.npy  ...     (tableaux 0/1, 2048 bits, forme 8x256 ou à plat)
    002/  ...
  finger/
    001/  s1.npy  s2.npy  ...     (tableaux 0/1, 1024 bits)
    002/  ...
```

Les formats `.png` et `.bmp` binaires sont aussi acceptés. Le premier échantillon de chaque sujet sert à l'enrôlement, les suivants aux essais authentiques.

### 4. Lancer les expériences

```bash
python3 experiments.py --data templates --subjects 100 --impostors 20 --reps 30 --out results
```

### 5. Recopier les résultats dans le papier

| Sortie | Où la mettre dans le papier |
|---|---|
| `results/hd_distributions.pdf` | Copier dans `paper/figures/` : la figure apparaît automatiquement |
| `results/frr_vs_keylength.pdf` | Copier dans `paper/figures/` : la figure apparaît automatiquement |
| `results/tables.tex` | Lignes prêtes à coller dans les tableaux « Key Regeneration Accuracy », « Computation Time » et « Decodability Cross-Matching Attack » |
| `results/accuracy.csv` | Chiffres à citer dans le texte (marqués `\TBD{}` en rouge) |

Les endroits à compléter sont en **rouge** dans le PDF : commande `\TBD{...}` dans `main.tex`.

## Paramètres utiles (`Params` dans `bio_key.py`)

- `rs_k` : taille du secret en symboles de 7 bits. La longueur de clé vaut 7 × k bits. Une valeur plus grande donne une clé plus longue mais un FRR plus élevé.
- `max_shift` : plage de recherche des rotations de l'iris (en colonnes).
- `scrypt_n` : coût du durcissement du PIN (2^15 par défaut).
- `permute` : permutation des bits dépendant du PIN et de l'époque, qui assure la révocabilité et la non-liaison.

## Tests réels déjà faits : empreintes FVC2002 (set B)

Fichiers ajoutés :
- `mbiochain/fp_features.py` : segmentation, amélioration par filtres de Gabor, squelette, minuties.
- `mbiochain/fp_align.py` : alignement par champ d'orientation et vecteur binaire de 1024 bits.
- `experiments_fvc.py` : protocole FVC.

```bash
python3 experiments_fvc.py --fvc DOSSIER_CONTENANT_DB1_B..DB4_B --out results_fvc
```

Résultats (`results_fvc/`, 40 doigts × 8 impressions, enrôlement sur l'impression 1, imposteurs pris dans la même base) :

| Mesure | Valeur |
|---|---|
| Distance de Hamming moyenne, authentiques / imposteurs | 0,179 / 0,296 |
| EER du vecteur binaire seul | 21,7 % |
| Fuzzy commitment RS(16,4), clé de 28 bits | FRR 3,2 %, FAR 65,6 % (le PIN est supposé connu de l'attaquant) |
| Fuzzy commitment RS(16,8), clé de 56 bits | FRR 4,3 %, FAR 56,7 % |
| Temps d'extraction / d'alignement | 72 ms / 18 ms par image |

**Conclusion :** l'empreinte seule ne suffit pas à protéger une clé, ce qui justifie l'ajout de l'iris. Il faut maintenant une base d'iris (par exemple CASIA-IrisV4) pour les tests multimodaux.
