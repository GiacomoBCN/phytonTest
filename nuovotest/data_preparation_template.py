# =============================================================================
# DATA PREPARATION TEMPLATE - REGRESSIONE & CLASSIFICAZIONE
# =============================================================================
# Template riutilizzabile per qualsiasi dataset CSV
# Compatibile sia con problemi di regressione che classificazione
# 
# ISTRUZIONI:
# 1. Modifica la sezione CONFIGURAZIONE
# 2. Esegui le celle in ordine
# 3. Il codice si adatta automaticamente al tuo dataset
# =============================================================================

# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Sklearn (opzionale, per encoding avanzato)
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# 2. CONFIGURAZIONE - MODIFICA QUESTA SEZIONE
# =============================================================================

# --- PERCORSO DATASET ---
DATASET_PATH = 'house_pricing.csv'  # TODO: Modificare con il tuo CSV
SEPARATOR = ','  # Separatore CSV (di solito ',' o ';')

# --- INFORMAZIONI DATASET ---
TARGET_COLUMN = 'SalePrice'  # Nome colonna target
TASK_TYPE = 'regression'  # 'regression' o 'classification'

# --- COLONNE DA RIMUOVERE ---
# Colonne inutili (ID, split, timestamp, ecc.)
COLS_TO_REMOVE = ['Split', 'Id']  # TODO: Aggiungere colonne da eliminare

# --- GESTIONE MISSING VALUES ---
# Soglia per rimuovere colonne con troppi NaN (es: 0.8 = rimuovi se >80% NaN)
MISSING_THRESHOLD = 0.8

# --- OUTLIERS (solo per regressione) ---
HANDLE_OUTLIERS = True  # True/False
OUTLIER_METHOD = 'IQR'  # 'IQR', 'zscore', o 'percentile'
OUTLIER_THRESHOLD = 1.5  # Per IQR method

# --- CORRELAZIONE ---
CORRELATION_THRESHOLD = 0.9  # Rimuovi features con corr > questo valore

# --- OUTPUT ---
OUTPUT_PATH = 'data_prepared.csv'  # Nome file output


# =============================================================================
# 3. FUNZIONI HELPER
# =============================================================================

def print_section(title):
    """Stampa intestazione sezione"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_subsection(title):
    """Stampa intestazione sottosezione"""
    print("\n" + "-"*70)
    print(f"{title}")
    print("-"*70)


def analyze_missing(df, show_details=True):
    """
    Analizza missing values nel dataframe
    
    Args:
        df: pandas DataFrame
        show_details: se True, mostra dettagli per colonna
    
    Returns:
        DataFrame con statistiche missing per colonna
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_pct': (df.isna().sum().values / len(df) * 100).round(2)
    })
    
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )
    
    if show_details and len(missing_stats) > 0:
        print("\nColonne con missing values:")
        for _, row in missing_stats.iterrows():
            print(f"  {row['column']:25s}: {row['missing_count']:5.0f} ({row['missing_pct']:5.1f}%)")
    
    return missing_stats


def detect_column_type(df, col):
    """
    Detecta automaticamente il tipo di colonna
    
    Returns:
        'numeric', 'categorical_binary', 'categorical_ordinal', 'categorical_nominal'
    """
    # Numerica
    if pd.api.types.is_numeric_dtype(df[col]):
        unique_values = df[col].nunique()
        if unique_values == 2:
            return 'numeric_binary'
        elif unique_values < 10:
            return 'numeric_discrete'
        else:
            return 'numeric_continuous'
    
    # Categorica
    else:
        unique_values = df[col].nunique()
        if unique_values == 2:
            return 'categorical_binary'
        elif unique_values <= 5:
            return 'categorical_low_cardinality'
        else:
            return 'categorical_high_cardinality'


def impute_missing(df, strategy='auto'):
    """
    Imputa missing values automaticamente
    
    Args:
        df: pandas DataFrame
        strategy: 'auto', 'mean', 'median', 'mode', 'drop'
    
    Returns:
        DataFrame con missing imputati
    """
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if df_imputed[col].isna().sum() == 0:
            continue
        
        col_type = detect_column_type(df_imputed, col)
        
        # Strategia automatica basata sul tipo
        if strategy == 'auto':
            if 'numeric' in col_type:
                # Usa mediana per numeriche (più robusto agli outlier)
                fill_value = df_imputed[col].median()
                df_imputed[col].fillna(fill_value, inplace=True)
                print(f"  ✓ {col:25s}: imputato con MEDIANA ({fill_value:.2f})")
            
            elif 'categorical' in col_type:
                # Usa moda per categoriche
                fill_value = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'unknown'
                df_imputed[col].fillna(fill_value, inplace=True)
                print(f"  ✓ {col:25s}: imputato con MODA ({fill_value})")
        
        # Strategie manuali
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        elif strategy == 'mode':
            df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
    return df_imputed


def remove_outliers_iqr(df, columns, threshold=1.5):
    """
    Rimuove outliers usando IQR method
    
    Args:
        df: pandas DataFrame
        columns: lista di colonne numeriche
        threshold: moltiplicatore IQR (default 1.5)
    
    Returns:
        DataFrame senza outliers
    """
    df_clean = df.copy()
    removed_total = 0
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Conta outliers
        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            # Rimuovi righe con outliers
            df_clean = df_clean[~outliers_mask]
            removed_total += n_outliers
            print(f"  ✓ {col:25s}: rimossi {n_outliers:4d} outliers")
    
    if removed_total > 0:
        print(f"\n  Total rows removed: {removed_total}")
        print(f"  Remaining rows: {len(df_clean)}")
    
    return df_clean


def encode_categorical(df, method='auto'):
    """
    Encoding automatico variabili categoriche
    
    Args:
        df: pandas DataFrame
        method: 'auto', 'onehot', 'label'
    
    Returns:
        DataFrame con variabili encodate
    """
    df_encoded = df.copy()
    
    # Identifica colonne categoriche
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        col_type = detect_column_type(df_encoded, col)
        
        if method == 'auto':
            # Binary: converti in 0/1
            if col_type == 'categorical_binary':
                unique_vals = df_encoded[col].dropna().unique()
                if len(unique_vals) == 2:
                    df_encoded[col] = (df_encoded[col] == unique_vals[1]).astype(int)
                    print(f"  ✓ {col:25s}: BINARY encoding ({unique_vals[0]}=0, {unique_vals[1]}=1)")
            
            # Low cardinality: One-Hot encoding
            elif col_type == 'categorical_low_cardinality':
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                print(f"  ✓ {col:25s}: ONE-HOT encoding ({len(dummies.columns)} dummies)")
            
            # High cardinality: Label encoding (o frequency)
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                print(f"  ✓ {col:25s}: LABEL encoding ({df_encoded[col].nunique()} categories)")
    
    return df_encoded


def check_high_correlation(df, threshold=0.9, exclude_cols=None):
    """
    Identifica features altamente correlate
    
    Args:
        df: pandas DataFrame
        threshold: soglia correlazione
        exclude_cols: colonne da escludere (es: target)
    
    Returns:
        Lista di colonne da rimuovere
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Seleziona solo colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        return []
    
    # Calcola correlazione
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Trova coppie altamente correlate
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr = corr_matrix.where(upper_triangle)
    
    # Identifica colonne da rimuovere
    to_drop = set()
    
    for col in high_corr.columns:
        correlated_features = high_corr.index[high_corr[col] > threshold].tolist()
        if correlated_features:
            # Rimuovi la feature con correlazione più alta
            for feat in correlated_features:
                corr_value = high_corr.loc[feat, col]
                print(f"  ⚠️ {col} <-> {feat}: {corr_value:.3f}")
                to_drop.add(feat)
    
    return list(to_drop)


# =============================================================================
# 4. LOAD DATASET
# =============================================================================

print_section("1. LOAD DATASET")

df = pd.read_csv(DATASET_PATH, sep=SEPARATOR)

print(f"\n✓ Dataset loaded: {DATASET_PATH}")
print(f"  Shape: {df.shape} (rows x columns)")
print(f"  Task type: {TASK_TYPE.upper()}")
print(f"  Target: {TARGET_COLUMN}")

print("\nFirst 3 rows:")
print(df.head(3))

print("\nData types:")
print(df.dtypes.value_counts())


# =============================================================================
# 5. INITIAL INSPECTION
# =============================================================================

print_section("2. INITIAL INSPECTION")

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"✓ Removed {duplicates} duplicate rows")

# Basic stats
print("\nNumerical features summary:")
print(df.describe())


# =============================================================================
# 6. REMOVE UNNECESSARY COLUMNS
# =============================================================================

print_section("3. REMOVE UNNECESSARY COLUMNS")

if COLS_TO_REMOVE:
    # Verifica che esistano
    cols_existing = [col for col in COLS_TO_REMOVE if col in df.columns]
    
    if cols_existing:
        df.drop(columns=cols_existing, inplace=True)
        print(f"\n✓ Removed columns: {cols_existing}")
        print(f"  New shape: {df.shape}")
    else:
        print(f"\n⚠️ No columns to remove (not found in dataset)")
else:
    print("\n⚠️ No columns specified for removal")


# =============================================================================
# 7. MISSING VALUES ANALYSIS
# =============================================================================

print_section("4. MISSING VALUES ANALYSIS")

missing_stats = analyze_missing(df, show_details=True)

if len(missing_stats) == 0:
    print("\n✓ No missing values detected!")
else:
    print(f"\n  Total missing values: {df.isna().sum().sum()}")
    print(f"  Affected columns: {len(missing_stats)}")


# =============================================================================
# 8. REMOVE HIGH-MISSING COLUMNS
# =============================================================================

print_subsection("4.1 Remove columns with too many missing values")

# Trova colonne con troppi missing (escludendo il target)
cols_to_check = [col for col in df.columns if col != TARGET_COLUMN]
high_missing_cols = []

for col in cols_to_check:
    missing_pct = df[col].isna().sum() / len(df)
    if missing_pct > MISSING_THRESHOLD:
        high_missing_cols.append((col, missing_pct))

if high_missing_cols:
    print(f"\nColumns with >{MISSING_THRESHOLD*100}% missing:")
    for col, pct in high_missing_cols:
        print(f"  {col:25s}: {pct*100:.1f}%")
    
    # Rimuovi
    cols_to_drop = [col for col, _ in high_missing_cols]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\n✓ Removed {len(cols_to_drop)} columns")
    print(f"  New shape: {df.shape}")
else:
    print("\n✓ No columns exceed the missing threshold")


# =============================================================================
# 9. IMPUTE MISSING VALUES
# =============================================================================

print_subsection("4.2 Impute missing values")

# Salva NaN del target (se presenti - test set)
target_nan_mask = df[TARGET_COLUMN].isna() if TARGET_COLUMN in df.columns else None

# Imputa features (escludendo target)
features_to_impute = [col for col in df.columns if col != TARGET_COLUMN]
df_features = df[features_to_impute].copy()

if df_features.isna().sum().sum() > 0:
    print("\nImputing missing values:")
    df[features_to_impute] = impute_missing(df_features, strategy='auto')
    
    # Verifica
    remaining_nan = df[features_to_impute].isna().sum().sum()
    if remaining_nan == 0:
        print("\n✓ All features imputed successfully!")
    else:
        print(f"\n⚠️ Still {remaining_nan} missing values in features")
else:
    print("\n✓ No missing values to impute in features")

# Ripristina NaN del target (se erano presenti)
if target_nan_mask is not None:
    df.loc[target_nan_mask, TARGET_COLUMN] = np.nan


# =============================================================================
# 10. OUTLIER DETECTION & REMOVAL (solo REGRESSIONE)
# =============================================================================

if TASK_TYPE == 'regression' and HANDLE_OUTLIERS:
    print_section("5. OUTLIER DETECTION & REMOVAL")
    
    # Seleziona colonne numeriche (escludendo target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
    
    if len(numeric_cols) > 0:
        print(f"\nRemoving outliers from {len(numeric_cols)} numeric features...")
        print(f"Method: {OUTLIER_METHOD} (threshold={OUTLIER_THRESHOLD})")
        
        initial_rows = len(df)
        df = remove_outliers_iqr(df, numeric_cols, threshold=OUTLIER_THRESHOLD)
        
        print(f"\n✓ Outlier removal complete")
        print(f"  Rows before: {initial_rows}")
        print(f"  Rows after: {len(df)}")
        print(f"  Removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
    else:
        print("\n⚠️ No numeric columns for outlier detection")


# =============================================================================
# 11. CATEGORICAL ENCODING
# =============================================================================

print_section("6. CATEGORICAL ENCODING")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]

if len(categorical_cols) > 0:
    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    df = encode_categorical(df, method='auto')
    print(f"\n✓ Encoding complete")
    print(f"  New shape: {df.shape}")
else:
    print("\n✓ No categorical features to encode")


# =============================================================================
# 12. CHECK HIGH CORRELATIONS
# =============================================================================

print_section("7. CHECK HIGH CORRELATIONS")

print(f"\nChecking correlations (threshold={CORRELATION_THRESHOLD})...")
cols_to_drop = check_high_correlation(df, threshold=CORRELATION_THRESHOLD, 
                                       exclude_cols=[TARGET_COLUMN])

if cols_to_drop:
    print(f"\n✓ Found {len(cols_to_drop)} highly correlated features to remove")
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  New shape: {df.shape}")
else:
    print("\n✓ No highly correlated features found")


# =============================================================================
# 13. FINAL VALIDATION
# =============================================================================

print_section("8. FINAL VALIDATION")

print("\nFinal dataset info:")
print(f"  Shape: {df.shape}")
print(f"  Missing in features: {df.drop(columns=[TARGET_COLUMN]).isna().sum().sum()}")
print(f"  Missing in target: {df[TARGET_COLUMN].isna().sum()} (test set)")

print("\nData types:")
print(df.dtypes.value_counts())

print("\nFinal summary statistics:")
print(df.describe())


# =============================================================================
# 14. SAVE PREPARED DATASET
# =============================================================================

print_section("9. SAVE PREPARED DATASET")

df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✓ Dataset saved: {OUTPUT_PATH}")
print(f"  Final shape: {df.shape}")
print(f"  Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"  Columns: {list(df.columns)}")

print("\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)
