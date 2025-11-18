# TABELLA STEP DATA PREPARATION

## Legenda File:
- **File 1**: `7_2_Data_preparation_bank_term_deposit_SOL.ipynb`
- **File 2**: `4_3_Data_preparation.ipynb` (house pricing - tuo lavoro)
- **File 3**: `4_3_Data_preparation_house_pricing_SOL.ipynb` (house pricing - soluzione)
- **File 4**: `Bank_Data_preparation.ipynb` (tuo lavoro bank)

---

| # | STEP | DESCRIZIONE | PRESENTE IN |
|---|------|-------------|-------------|
| 1 | **Imports** | Importare librerie necessarie (pandas, numpy, matplotlib, seaborn, sklearn) | File 1, 2, 3, 4 |
| 2 | **Load Dataset** | Caricare il CSV con pd.read_csv() | File 1, 2, 3, 4 |
| 3 | **Select Data** | Decidere quali dataset/features usare (opzionale) | File 3, 4 |
| 4 | **Remove Unnecessary Features** | Eliminare colonne inutili/ridondanti (ID, split, ecc.) | File 1, 4 |
| 5 | **Clean Data - Sostituire Valori Errati** | Sostituire valori come '?' con 'unknown' o valori corretti | File 4 |
| 6 | **Handle Missing Values - Identificazione** | Identificare NaN/missing values per colonna | File 1, 2, 3, 4 |
| 7 | **Handle Missing Values - Imputazione** | Imputare missing con media/mediana/moda/strategia custom | File 1, 2, 3, 4 |
| 8 | **Handle Missing Values - Rimozione** | Rimuovere righe/colonne con troppi missing | File 1, 3 |
| 9 | **Outlier Detection** | Identificare outlier con IQR, z-score, o visualizzazioni | File 1 |
| 10 | **Outlier Treatment** | Gestire outlier (cap, remove, transform) | File 1 |
| 11 | **Encoding - Label Encoding** | Encoding ordinale per variabili categoriche ordinate | File 2, 3, 4 |
| 12 | **Encoding - One-Hot Encoding** | Creare dummy variables per categoriche nominali | File 1, 2, 3, 4 |
| 13 | **Encoding - Target Encoding** | Encoding basato sul target (media target per categoria) | File 1 |
| 14 | **Encoding - Frequency Encoding** | Encoding basato sulla frequenza della categoria | File 1 |
| 15 | **Feature Engineering - Binning** | Creare categorie/gruppi da variabili continue (age_group, ecc.) | File 1, 4 |
| 16 | **Feature Engineering - Interazioni** | Creare feature combinando altre (ratio, prodotto, ecc.) | File 1 |
| 17 | **Feature Engineering - Date Features** | Estrarre componenti da date (giorno, mese, stagione) | File 1, 4 |
| 18 | **Feature Engineering - Cyclical Features** | Trasformare variabili cicliche con sin/cos (mese, giorno) | File 1 |
| 19 | **Feature Engineering - Boolean Features** | Creare flag binari (has_X, is_Y) | File 2, 4 |
| 20 | **Feature Engineering - Aggregazioni** | Creare feature aggregate (count, sum, mean per gruppo) | File 1 |
| 21 | **Feature Scaling/Normalization** | Standardizzare o normalizzare features numeriche | File 1 |
| 22 | **Check Correlations** | Calcolare matrice di correlazione tra features | File 1, 4 |
| 23 | **Remove Highly Correlated Features** | Rimuovere feature con correlazione > soglia (0.8-0.9) | File 1, 3 |
| 24 | **Train-Test Split** | Separare train e test set (se non già fatto) | File 1 |
| 25 | **Target Encoding (sul train)** | Applicare target encoding usando SOLO train set | File 1 |
| 26 | **Convert Target to Numeric** | Convertire target da categorico a numerico (yes/no → 1/0) | File 4 |
| 27 | **Final Data Validation** | Verificare dtypes, shape, missing values finali | File 1, 2, 4 |
| 28 | **Save Prepared Dataset** | Salvare il dataframe processato in CSV | File 1, 2, 3, 4 |

---

## NOTE IMPORTANTI:

### ORDINE CRITICO:
1. **Missing Values** → va fatto PRIMA di encoding e feature engineering
2. **Outlier** → dopo missing, prima di scaling
3. **Train/Test Split** → PRIMA di scaling e target encoding (per evitare data leakage)
4. **Encoding ordinale** → quando la variabile ha ordine naturale (low < medium < high)
5. **One-Hot** → quando la variabile NON ha ordine (colori, categorie)
6. **Target Encoding** → SOLO dopo split, usando train set
7. **Scaling** → ultimo step prima del salvataggio
8. **Correlations** → controllare DOPO feature engineering

### DECISIONI CONTEXT-DEPENDENT:
- **Binning**: dipende dal dataset e dalla variabile
- **Cyclical encoding**: solo per variabili cicliche (mese, ora, giorno settimana)
- **Outlier removal**: dipende dal problema e dal modello
- **Feature selection**: dipende da correlazioni e importanza

### DATA LEAKAGE - ATTENZIONE:
- Target encoding: calcolare SOLO su train, applicare a test
- Scaling: fit SOLO su train, transform su test
- Imputazione: strategia può essere calcolata su train

---

## STEP MINIMI PER UN TEMPLATE GENERICO:
1. Load dataset
2. Identify and handle missing values
3. Encode categorical variables
4. Feature engineering (opzionale ma consigliato)
5. Check correlations
6. Save prepared dataset

**Per esame**: prepara funzioni riutilizzabili per ogni step!
