# =============================================================================
# ESEMPI DI UTILIZZO - DATA PREPARATION TEMPLATE
# =============================================================================

# =============================================================================
# ESEMPIO 1: HOUSE PRICING (REGRESSIONE)
# =============================================================================

DATASET_PATH = 'house_pricing.csv'
SEPARATOR = ','
TARGET_COLUMN = 'SalePrice'
TASK_TYPE = 'regression'
COLS_TO_REMOVE = ['Split', 'Id']
MISSING_THRESHOLD = 0.8
HANDLE_OUTLIERS = True
OUTLIER_METHOD = 'IQR'
OUTLIER_THRESHOLD = 1.5
CORRELATION_THRESHOLD = 0.9
OUTPUT_PATH = 'house_pricing_prepared.csv'


# =============================================================================
# ESEMPIO 2: BANK MARKETING (CLASSIFICAZIONE)
# =============================================================================

DATASET_PATH = 'data/bank/bank_term_deposit.csv'
SEPARATOR = ','
TARGET_COLUMN = 'y'
TASK_TYPE = 'classification'
COLS_TO_REMOVE = ['id', 'split']
MISSING_THRESHOLD = 0.8
HANDLE_OUTLIERS = False  # Non rimuovere outliers in classificazione
OUTLIER_METHOD = 'IQR'
OUTLIER_THRESHOLD = 1.5
CORRELATION_THRESHOLD = 0.95  # Più permissivo per classificazione
OUTPUT_PATH = 'bank_prepared.csv'


# =============================================================================
# ESEMPIO 3: DATASET SCONOSCIUTO (APPROCCIO CONSERVATIVO)
# =============================================================================

DATASET_PATH = 'unknown_dataset.csv'
SEPARATOR = ','
TARGET_COLUMN = 'target'  # TODO: Verificare nome corretto
TASK_TYPE = 'regression'  # TODO: Cambiare se necessario
COLS_TO_REMOVE = []  # Non rimuovere nulla finché non ispeziono
MISSING_THRESHOLD = 0.9  # Più permissivo
HANDLE_OUTLIERS = False  # Disabilitato per sicurezza
OUTLIER_METHOD = 'IQR'
OUTLIER_THRESHOLD = 1.5
CORRELATION_THRESHOLD = 0.95  # Più permissivo
OUTPUT_PATH = 'data_prepared.csv'


# =============================================================================
# WORKFLOW CONSIGLIATO PER ESAME
# =============================================================================

"""
1. APRI IL TEMPLATE
   - Copia data_preparation_template.py

2. MODIFICA CONFIGURAZIONE
   - Imposta DATASET_PATH, TARGET_COLUMN, TASK_TYPE
   - Identifica COLS_TO_REMOVE (guardando df.head())

3. ESEGUI IL TEMPLATE
   - python data_preparation_template.py
   
4. VERIFICA OUTPUT
   - Controlla il CSV generato
   - Se serve, aggiusta parametri e ri-esegui

5. PASSA A ML_EXECUTION
   - Carica il CSV preparato
   - Train/test split
   - Pipeline con scaling
   - Training e evaluation
"""


# =============================================================================
# PARAMETRI CONSIGLIATI PER TIPO DI PROBLEMA
# =============================================================================

# REGRESSIONE
regressione_config = {
    'MISSING_THRESHOLD': 0.8,      # Rimuovi colonne con >80% missing
    'HANDLE_OUTLIERS': True,       # Rimuovi outliers
    'OUTLIER_THRESHOLD': 1.5,      # IQR * 1.5 (standard)
    'CORRELATION_THRESHOLD': 0.9,  # Rimuovi se corr > 0.9
}

# CLASSIFICAZIONE
classificazione_config = {
    'MISSING_THRESHOLD': 0.8,      # Rimuovi colonne con >80% missing
    'HANDLE_OUTLIERS': False,      # Non rimuovere outliers
    'OUTLIER_THRESHOLD': 1.5,      # (non usato)
    'CORRELATION_THRESHOLD': 0.95, # Più permissivo (0.95)
}


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
PROBLEMA: "Too many missing values still present"
SOLUZIONE: Abbassa MISSING_THRESHOLD (es: 0.7 invece di 0.8)

PROBLEMA: "Categorical encoding failed"
SOLUZIONE: Verifica che non ci siano categorie strane (spazi, caratteri speciali)

PROBLEMA: "Too many rows removed by outlier detection"
SOLUZIONE: Aumenta OUTLIER_THRESHOLD (es: 2.0 o 3.0) o disabilita (False)

PROBLEMA: "High correlation not reducing features"
SOLUZIONE: Abbassa CORRELATION_THRESHOLD (es: 0.85 invece di 0.9)

PROBLEMA: "Target has missing values in train set"
SOLUZIONE: Controlla che TARGET_COLUMN sia corretto, i NaN devono essere solo nel test
"""
