# RIASSUNTO NOTEBOOK: 5.4.ML_example.ipynb

## 📚 SEZIONE 1: Importazione Librerie
**Celle:** 1-3 (1 markdown + 1 code)
**Cosa fa:** Importa tutte le librerie Python necessarie per il progetto di Machine Learning.
**Perché è importante:** Le librerie sono come "cassette degli attrezzi" che contengono funzioni già pronte. Senza di esse, dovremmo scrivere tutto il codice da zero (che richiederebbe mesi!).
**Concetti chiave:**
- **NumPy** (`numpy`): per fare calcoli matematici su array (liste di numeri)
- **Pandas** (`pandas`): per lavorare con tabelle di dati (come Excel, ma in Python)
- **Matplotlib** (`matplotlib`): per creare grafici e visualizzazioni
- **Scikit-learn** (`sklearn`): la libreria principale per Machine Learning, contiene tutti i modelli e strumenti
- **SciPy** (`scipy`): per statistiche avanzate e distribuzioni di probabilità

**Analogia:** Immagina di dover costruire una casa. Invece di creare da zero martello, chiodi e seghe, usi strumenti già pronti. Le librerie sono questi strumenti pronti all'uso.

---

## 📚 SEZIONE 2: Caricamento e Visualizzazione Dati
**Celle:** 4-9 (2 markdown + 3 code)
**Cosa fa:** Carica il dataset delle case da un file CSV e lo visualizza per capire con cosa stiamo lavorando.
**Perché è importante:** Prima di costruire un modello, dobbiamo capire i nostri dati: quante righe (case) abbiamo? Quante colonne (caratteristiche)?
**Concetti chiave:**
- **CSV (Comma Separated Values)**: un tipo di file dove i dati sono separati da virgole, come un foglio Excel
- **DataFrame** (`df`): una tabella di dati in Pandas (righe e colonne)
- **Shape**: dimensioni del dataset (numero_righe, numero_colonne)
- **Split**: colonna che indica se i dati sono per training ("labeled") o per predizioni finali ("leaderboard")
- **SalePrice**: la colonna che contiene il prezzo di vendita delle case (questo è ciò che vogliamo predire!)

**Quello che succede:**
1. Leggiamo il file `gb_house_pricing.csv`
2. Visualizziamo quante righe e colonne contiene
3. Separiamo i dati "labeled" (con prezzo) da quelli "leaderboard" (senza prezzo, da predire)

---

## 📚 SEZIONE 3: Separazione Train/Test e Features/Target
**Celle:** 10-12 (1 markdown + 2 code)
**Cosa fa:** Divide i dati in due gruppi: uno per "insegnare" al modello (train) e uno per testarlo (test). Poi separa le "caratteristiche" (X) dal "target" (y).
**Perché è importante:** È come studiare per un esame: studi su alcuni problemi (train) e poi ti testi su problemi nuovi (test) per vedere se hai davvero imparato.
**Concetti chiave:**
- **Features (X)**: le caratteristiche delle case (numero stanze, metri quadri, anno costruzione, ecc.)
- **Target (y)**: ciò che vogliamo predire (il prezzo di vendita, `SalePrice`)
- **Train/Test Split**: divisione dei dati in 80% training e 20% test
- **X_train, X_test**: features per training e test
- **y_train, y_test**: target (prezzi) per training e test

**Analogia:** Immagina di voler imparare a riconoscere i gatti. Guardi 80 foto di gatti con l'etichetta "gatto" (training), poi ti mostro 20 foto nuove e devi dire se sono gatti (test). Se indovini 18 su 20, hai imparato bene!

---

## 📚 SEZIONE 4: Cross-Validation e Metriche di Valutazione
**Celle:** 13-15 (1 markdown + 2 code)
**Cosa fa:** Configura la cross-validation (un modo più robusto di testare il modello) e definisce le metriche per misurare quanto è bravo il modello.
**Perché è importante:** Vogliamo essere sicuri che il modello funzioni bene, non solo per fortuna. La cross-validation testa il modello in modi diversi.
**Concetti chiave:**
- **Cross-Validation (K-Fold con K=10)**: divide i dati in 10 "pieghe" (folds)
  - Usa 9 pieghe per training e 1 per test
  - Ripete 10 volte, cambiando ogni volta la piega di test
  - Alla fine fa la media dei risultati
- **Metriche di errore** (più basse = meglio):
  - **MSE (Mean Squared Error)**: errore quadratico medio. Penalizza molto gli errori grandi. Difficile da interpretare perché è al quadrato.
  - **RMSE (Root Mean Squared Error)**: radice quadrata del MSE. Più facile da capire perché ha la stessa unità del target (es: euro).
  - **MAE (Mean Absolute Error)**: errore assoluto medio. Più facile da interpretare: "in media sbaglio di X euro".

**Analogia:** Invece di fare un solo esame, fai 10 esami diversi e calcoli la media. Così sai se sei davvero bravo o se hai avuto solo fortuna una volta.

---

## 📚 SEZIONE 5: Modello Baseline (DummyRegressor)
**Celle:** 16-17 (1 markdown + 1 code)
**Cosa fa:** Crea un modello "stupido" che predice sempre la media dei prezzi del training set.
**Perché è importante:** Serve come punto di partenza. Se i nostri modelli "intelligenti" non battono questo modello stupido, c'è qualcosa che non va!
**Concetti chiave:**
- **Baseline**: un modello semplice usato come riferimento
- **DummyRegressor con strategy='mean'**: predice sempre lo stesso valore (la media)
- **Sanity check**: un controllo di base per assicurarci che il resto del progetto abbia senso

**Analogia:** Se ti chiedo di predire il prezzo di una casa e non sai niente, la strategia più semplice è dire "la media di tutte le case che conosco". Questo è il baseline.

---

## 📚 SEZIONE 6: Regressione Lineare
**Celle:** 18-19 (1 markdown + 1 code)
**Cosa fa:** Prova un modello di regressione lineare, che cerca di trovare una linea (o iperpiano) che si adatta ai dati.
**Perché è importante:** La regressione lineare è il modello più semplice dopo il baseline. Se funziona bene, vuol dire che i dati hanno una relazione lineare.
**Concetti chiave:**
- **Regressione Lineare**: cerca una relazione lineare tra features e target (es: più metri quadri = prezzo più alto)
- **Pipeline**: una sequenza di passaggi applicati in ordine
- **StandardScaler**: normalizza i dati (media=0, deviazione standard=1). Importante perché le features hanno scale diverse (es: metri quadri vs numero stanze)
- **Pipeline = Scaler + LinearRegression**: prima normalizza, poi applica la regressione

**Analogia:** Immagina di tracciare una linea retta su un grafico di punti. La regressione lineare cerca la linea che passa il più vicino possibile a tutti i punti.

---

## 📚 SEZIONE 7: Hyperparameter Tuning - Decision Tree
**Celle:** 20-27 (3 markdown + 2 code)
**Cosa fa:** Usa RandomizedSearchCV per trovare i migliori "iperparametri" (impostazioni) per un Decision Tree (albero decisionale).
**Perché è importante:** I modelli hanno molte "manopole" da regolare. RandomizedSearchCV prova combinazioni casuali per trovare le migliori.
**Concetti chiave:**
- **Decision Tree (Albero Decisionale)**: un modello che fa domande in sequenza (es: "La casa ha più di 100 mq? Sì → ha più di 3 stanze? No → prezzo stimato: 200k")
- **Iperparametri**: impostazioni del modello che NON vengono apprese dai dati ma devono essere scelte da noi:
  - `max_depth`: quanto è profondo l'albero (quante domande può fare)
  - `min_samples_split`: quanti dati servono per fare una nuova domanda
  - `min_samples_leaf`: quanti dati devono finire in ogni "foglia" finale
- **RandomizedSearchCV**: prova N combinazioni casuali di iperparametri e sceglie la migliore
- **Overfitting**: quando il modello "impara a memoria" i dati di training ma non funziona su dati nuovi

**Analogia:** È come regolare il volume, i bassi e gli alti di uno stereo. Devi trovare la combinazione giusta. RandomizedSearchCV prova tante combinazioni e sceglie quella che suona meglio.

---

## 📚 SEZIONE 8: Hyperparameter Tuning - Gradient Boosting (Principale)
**Celle:** 33-49 (4 markdown + 8 code)
**Cosa fa:** Usa RandomizedSearchCV per ottimizzare un GradientBoostingRegressor, un modello molto potente che combina tanti alberi decisionali "deboli".
**Perché è importante:** Gradient Boosting è uno dei migliori algoritmi per problemi di regressione. Questa sezione cerca la migliore configurazione possibile.
**Concetti chiave:**
- **Gradient Boosting**: tecnica che combina molti alberi decisionali "deboli" in sequenza. Ogni albero cerca di correggere gli errori dell'albero precedente.
- **Ensemble**: combinazione di più modelli per ottenere un modello più forte
- **Iperparametri principali**:
  - `n_estimators`: quanti alberi usare (più alberi = modello più complesso)
  - `learning_rate`: quanto velocemente il modello "impara" (troppo alto = instabile, troppo basso = lento)
  - `max_depth`: profondità di ogni albero
  - `subsample`: percentuale di dati usata per ogni albero (aiuta a prevenire overfitting)
- **Pipeline con ColumnTransformer**: gestisce features numeriche e categoriche separatamente
- **Iterazioni multiple**: il notebook mostra varie prove per trovare i migliori 5 modelli

**Analogia:** Immagina una classe dove ogni studente (albero) prova a risolvere un problema. Il primo studente sbaglia un po', il secondo guarda gli errori del primo e cerca di correggerli, il terzo corregge gli errori del secondo, ecc. Alla fine, la risposta collettiva è molto migliore di quella di un singolo studente.

---

## 📚 SEZIONE 9: Predizioni Finali e Creazione Submission
**Celle:** 50-58 (2 markdown + 7 code)
**Cosa fa:** Usa il miglior modello trovato per fare predizioni sul set "leaderboard" (case senza prezzo) e crea un file CSV con le predizioni.
**Perché è importante:** Questo è l'obiettivo finale: usare il modello addestrato per predire prezzi di case nuove, mai viste prima.
**Concetti chiave:**
- **Leaderboard test set**: dati senza il target (SalePrice), da usare per predizioni finali
- **best_estimator_**: il miglior modello trovato da RandomizedSearchCV, già addestrato
- **predict()**: funzione che fa predizioni su nuovi dati
- **Submission file**: file CSV con formato `id, prediction` per sottomettere le predizioni
- **Range ID specifico**: in questo caso gli ID vanno da 795 a 992 (specifico del dataset)

**Processo:**
1. Prepara i dati di leaderboard (rimuove colonne inutili)
2. Usa `gbr_rscv.best_estimator_.predict()` per fare predizioni
3. Crea un DataFrame con ID e predizioni
4. Salva tutto in `submission.csv`

**Analogia:** Dopo aver studiato tanto (training), finalmente fai l'esame vero (predizioni sul leaderboard) e consegni il compito (submission file).

---

## 🎯 OBIETTIVO FINALE

Questo notebook costruisce un sistema di Machine Learning per **predire il prezzo di vendita delle case** basandosi sulle loro caratteristiche (metri quadri, numero stanze, anno costruzione, ecc.). Partendo da un modello semplice (baseline), passa attraverso modelli via via più complessi (regressione lineare, alberi decisionali, gradient boosting), ottimizzando gli iperparametri per trovare la migliore configurazione possibile. Alla fine, usa il miglior modello per predire i prezzi di case nuove e crea un file di submission pronto per essere valutato.

---

## 📊 FLUSSO COMPLETO

```
📁 CARICAMENTO DATI
   ↓
   gb_house_pricing.csv
   ↓
🔀 SEPARAZIONE DATI
   ↓
   ├─→ labeled (con SalePrice) → Train/Test
   └─→ leaderboard (senza SalePrice) → Per predizioni finali
   ↓
🎯 SEPARAZIONE FEATURES/TARGET
   ↓
   X (caratteristiche case) e y (prezzi)
   ↓
📊 SETUP CROSS-VALIDATION + METRICHE
   ↓
   KFold (10 pieghe) + MSE/RMSE/MAE
   ↓
🤖 MODELLO BASELINE
   ↓
   DummyRegressor (predice sempre la media)
   ↓
📈 REGRESSIONE LINEARE
   ↓
   Pipeline: StandardScaler → LinearRegression
   ↓
🌳 DECISION TREE con HYPERPARAMETER TUNING
   ↓
   RandomizedSearchCV per trovare migliori parametri
   ↓
🚀 GRADIENT BOOSTING con HYPERPARAMETER TUNING
   ↓
   RandomizedSearchCV → Trova miglior modello
   ↓
   ├─→ Analisi top 5 modelli
   ├─→ Secondo round di ottimizzazione
   └─→ Pipeline con preprocessing
   ↓
🎯 PREDIZIONI FINALI
   ↓
   best_estimator_.predict(leaderboard_test)
   ↓
💾 CREAZIONE SUBMISSION FILE
   ↓
   submission.csv (id, prediction)
   ↓
✅ PRONTO PER VALUTAZIONE!
```

---

## 📖 GLOSSARIO TERMINI TECNICI

- **Dataset**: insieme di dati organizzati in tabella
- **Features (X)**: le caratteristiche/variabili indipendenti usate per fare predizioni
- **Target (y)**: la variabile che vogliamo predire (in questo caso, SalePrice)
- **Training**: processo di "insegnamento" al modello usando dati con etichetta
- **Test**: valutazione del modello su dati nuovi per vedere se ha davvero imparato
- **Overfitting**: quando il modello "impara a memoria" i dati di training ma non funziona su dati nuovi
- **Cross-Validation**: tecnica per testare il modello in modo più robusto, dividendo i dati in K parti
- **Hyperparameters**: impostazioni del modello che devono essere scelte prima del training
- **Pipeline**: sequenza di operazioni applicate in ordine ai dati
- **Scaling/Normalizzazione**: processo per portare tutte le features sulla stessa scala

---

## 💡 CONSIGLI PER PRINCIPIANTI

1. **Leggi il notebook dall'inizio**: ogni sezione si basa sulla precedente
2. **Esegui una cella alla volta**: non correre, comprendi ogni passaggio
3. **Guarda i risultati**: dopo ogni cella, osserva cosa viene stampato
4. **Sperimenta**: prova a cambiare piccoli valori per vedere cosa succede
5. **Non preoccuparti se non capisci tutto subito**: il Machine Learning richiede pratica
6. **Focalizzati prima sul flusso generale**: poi approfondisci i dettagli

---

**Percorso di apprendimento consigliato:**
1. Prima lettura: capire il flusso generale
2. Seconda lettura: capire ogni sezione in dettaglio
3. Terza lettura: capire il codice riga per riga
4. Quarta volta: prova a modificare e sperimentare

Buono studio! 🎓
