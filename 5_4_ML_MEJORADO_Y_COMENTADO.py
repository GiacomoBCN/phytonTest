# ============================================
# GRADIENT BOOSTING REGRESSOR - CÓDIGO MEJORADO Y COMENTADO
# ============================================
# Este código busca los mejores hiperparámetros para GradientBoostingRegressor
# y evalúa los 5 mejores modelos en un segundo round

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================
# PASO 1: CONFIGURACIÓN DEL MODELO BASE
# ============================================
# Creamos el modelo GradientBoostingRegressor con random_state fijo
# para garantizar reproducibilidad de resultados
gbr = GradientBoostingRegressor(random_state=42)

# ============================================
# PASO 2: DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ============================================
# Definimos las distribuciones de valores a probar para cada hiperparámetro

param_dist = {
    # ---- FUNCIÓN DE PÉRDIDA ----
    # - squared_error: MSE estándar (mejor para datos sin outliers extremos)
    # - huber: Robusta a outliers (combinación de MSE y MAE)
    # - absolute_error: MAE (muy robusta pero más lenta)
    "loss": ["squared_error", "huber", "absolute_error"],
    
    # ---- LEARNING RATE Y NÚMERO DE ÁRBOLES ----
    # Learning rate bajo + más árboles = mejor generalización (pero más lento)
    # loguniform: explora valores logarítmicamente distribuidos
    "learning_rate": loguniform(1e-3, 3e-1),     # Rango: 0.001 - 0.3
    "n_estimators": randint(150, 1201),          # Rango: 150 - 1200 árboles
    
    # ---- STOCHASTIC GRADIENT BOOSTING ----
    # subsample < 1.0 usa una fracción aleatoria de datos en cada árbol
    # Ayuda a prevenir overfitting y acelera el entrenamiento
    "subsample": uniform(0.6, 0.4),              # Rango: 0.6 - 1.0
    
    # ---- ESTRUCTURA DEL ÁRBOL ----
    # Árboles poco profundos para datos con tendencia lineal
    "max_depth": randint(2, 5),                  # Rango: 2 - 4
    
    # Mínimo de muestras para dividir un nodo (como fracción del total)
    "min_samples_split": uniform(0.02, 0.18),    # Rango: 0.02 - 0.20 (2%-20%)
    
    # Mínimo de muestras en cada hoja (como fracción del total)
    "min_samples_leaf": uniform(0.01, 0.09),     # Rango: 0.01 - 0.10 (1%-10%)
    
    # Número de features a considerar en cada split
    # None = todas, "sqrt" = raíz cuadrada del total
    "max_features": [None, 1.0, 0.8, "sqrt"],
    
    # ---- CRITERIO DE SPLIT ----
    # friedman_mse: optimizado para gradient boosting (recomendado)
    # squared_error: MSE estándar
    "criterion": ["friedman_mse", "squared_error"],
    
    # ---- REGULARIZACIÓN ----
    # Reducción mínima de impureza requerida para hacer un split
    "min_impurity_decrease": uniform(0.0, 0.002),
    
    # Cost-complexity pruning alpha (mayor = más poda)
    "ccp_alpha": uniform(0.0, 0.01),
    
    # ---- PARÁMETRO ESPECÍFICO PARA HUBER ----
    # Solo relevante si loss='huber' o 'quantile'
    # Alpha determina el quantile (valores altos = más robusto)
    "alpha": uniform(0.85, 0.14),                # Rango: 0.85 - 0.99
    
    # ---- EARLY STOPPING (PARADA TEMPRANA) ----
    # Fracción de datos usada para validación interna
    "validation_fraction": uniform(0.1, 0.1),    # Rango: 0.10 - 0.20
    
    # Número de iteraciones sin mejora antes de parar
    "n_iter_no_change": randint(5, 16),          # Rango: 5 - 15
    
    # Tolerancia mínima de mejora requerida
    "tol": loguniform(1e-5, 1e-3),               # Rango: 0.00001 - 0.001
}

# ============================================
# PASO 3: DEFINICIÓN DE MÉTRICAS DE EVALUACIÓN
# ============================================
# Usamos scoring negativos porque sklearn maximiza los scores
# (valores menos negativos = mejor)
scoring = {
    "RMSE": "neg_root_mean_squared_error",  # Penaliza errores grandes
    "MAE": "neg_mean_absolute_error",       # Más robusto a outliers
}

# ============================================
# PASO 4: CONFIGURACIÓN DE RANDOMIZEDSEARCHCV
# ============================================
# RandomizedSearchCV prueba combinaciones aleatorias de hiperparámetros
# Es más eficiente que GridSearchCV cuando hay muchos parámetros

gbr_rscv = RandomizedSearchCV(
    gbr,                          # Modelo base
    param_distributions=param_dist,  # Espacio de búsqueda
    n_iter=80,                    # ✅ REDUCIDO de 150 a 80 (más rápido, igualmente efectivo)
    cv=kf,                        # KFold cross-validation (debe estar definido previamente)
    scoring=scoring,              # Métricas a calcular
    refit="RMSE",                 # Reentrena con el mejor modelo según RMSE
    n_jobs=-1,                    # Usa todos los cores del CPU
    verbose=1,                    # Muestra progreso
    random_state=42,              # Reproducibilidad
    return_train_score=True,      # Calcula scores en training también
)

# ============================================
# PASO 5: ENTRENAMIENTO (BÚSQUEDA DE HIPERPARÁMETROS)
# ============================================
print("🚀 Iniciando búsqueda de hiperparámetros...")
print(f"   - Probando {gbr_rscv.n_iter} combinaciones")
print(f"   - Con {kf.n_splits} folds de cross-validation")
print(f"   - Total de entrenamientos: {gbr_rscv.n_iter * kf.n_splits}")
print(f"   - Tiempo estimado: 2-3 minutos\n")

gbr_rscv.fit(X_train, y_train)

print("✅ Búsqueda completada!\n")

# ============================================
# PASO 6: MOSTRAR MEJORES RESULTADOS
# ============================================

# --- Tabla de mejores hiperparámetros ---
best_params = gbr_rscv.best_params_
best_params_table = pd.DataFrame(
    list(best_params.items()), 
    columns=['Hyperparameter', 'Best Value']
)

# --- Métricas del mejor modelo ---
train_rmse = -gbr_rscv.cv_results_['mean_train_RMSE'][gbr_rscv.best_index_]
val_rmse   = -gbr_rscv.cv_results_['mean_test_RMSE'][gbr_rscv.best_index_]
train_mae  = -gbr_rscv.cv_results_['mean_train_MAE'][gbr_rscv.best_index_]
val_mae    = -gbr_rscv.cv_results_['mean_test_MAE'][gbr_rscv.best_index_]

metrics_table = pd.DataFrame({
    'Metric': ['RMSE', 'MAE'],
    'Train': [round(train_rmse, 2), round(train_mae, 2)],
    'Validation': [round(val_rmse, 2), round(val_mae, 2)],
})
metrics_table['Difference'] = (metrics_table['Validation'] - metrics_table['Train']).round(2)
metrics_table['% Diff'] = ((metrics_table['Difference'] / metrics_table['Train']) * 100).round(2)

# --- Mostrar resultados ---
print("\n" + "="*60)
print(" MEJORES HIPERPARÁMETROS ENCONTRADOS")
print("="*60 + "\n")
print(best_params_table.to_string(index=False))

print("\n" + "="*60)
print(" RENDIMIENTO DEL MEJOR MODELO")
print("="*60 + "\n")
print(metrics_table.to_string(index=False))
print()

# ============================================
# PASO 7: ROUND 2 - EVALUACIÓN DE TOP 5 MODELOS
# ============================================
# Tomamos los 5 mejores modelos por MAE y los evaluamos en el test set
# Esto nos da una idea más clara del rendimiento real

print("\n" + "="*60)
print(" ROUND 2: EVALUACIÓN DE TOP 5 MODELOS")
print("="*60 + "\n")
print("Entrenando los 5 mejores modelos encontrados...")
print("(Esto puede tardar 1-2 minutos)\n")

# 1) Obtener los 5 mejores índices según MAE
cv_mae_neg = gbr_rscv.cv_results_['mean_test_MAE']
cv_rmse_neg = gbr_rscv.cv_results_.get('mean_test_RMSE', None)
cv_mae = -cv_mae_neg  # Convertir a positivo
cv_rmse = -cv_rmse_neg if cv_rmse_neg is not None else None

best_indices = np.argsort(cv_mae)[:5]  # Los 5 MAE más pequeños

# 2) Entrenar cada modelo y evaluar en train/test
round2_rows = []
for rank, i in enumerate(best_indices, start=1):
    params = gbr_rscv.cv_results_['params'][i]
    
    # Crear y entrenar modelo con estos parámetros
    model = GradientBoostingRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    
    # Predecir en train y test
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    
    # Calcular métricas
    mae_tr = mean_absolute_error(y_train, y_pred_tr)
    mae_te = mean_absolute_error(y_test, y_pred_te)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
    
    # Guardar resultados
    row = {
        "Rank": rank,
        "CV MAE": round(float(cv_mae[i]), 2),
        "CV RMSE": round(float(cv_rmse[i]), 2) if cv_rmse is not None else None,
        "Train MAE": round(mae_tr, 2),
        "Test MAE": round(mae_te, 2),
        "Train RMSE": round(rmse_tr, 2),
        "Test RMSE": round(rmse_te, 2),
        "Overfitting": "⚠️ Sí" if (mae_te - mae_tr) / mae_tr > 0.3 else "✅ No",
        "Params": params
    }
    round2_rows.append(row)

# 3) Crear tabla ordenada por Test MAE (mejor = menor)
round2_df = pd.DataFrame(round2_rows)
round2_df = round2_df.sort_values(by="Test MAE", ascending=True).reset_index(drop=True)

# Mostrar resultados (sin la columna Params para que sea más legible)
display_df = round2_df.drop('Params', axis=1)
print("🔁 Top 5 Modelos (ordenados por Test MAE):\n")
print(display_df.to_string(index=False))

# ============================================
# PASO 8: SELECCIONAR Y GUARDAR EL MEJOR MODELO
# ============================================
best_round2_params = round2_df.iloc[0]["Params"]
best_round2_model = GradientBoostingRegressor(random_state=42, **best_round2_params)
best_round2_model.fit(X_train, y_train)

print(f"\n✅ Mejor modelo del Round 2:")
print(f"   - Test MAE: {round2_df.iloc[0]['Test MAE']}")
print(f"   - Test RMSE: {round2_df.iloc[0]['Test RMSE']}")
print(f"   - Overfitting: {round2_df.iloc[0]['Overfitting']}")

# ============================================
# RESUMEN Y RECOMENDACIONES
# ============================================
print("\n" + "="*60)
print(" ANÁLISIS Y RECOMENDACIONES")
print("="*60 + "\n")

# Calcular diferencia promedio entre train y test
avg_train_mae = round2_df['Train MAE'].mean()
avg_test_mae = round2_df['Test MAE'].mean()
gap_pct = ((avg_test_mae - avg_train_mae) / avg_train_mae) * 100

print(f"📊 Brecha promedio Train-Test: {gap_pct:.1f}%")

if gap_pct < 20:
    print("   ✅ Excelente generalización")
elif gap_pct < 40:
    print("   ⚠️ Generalización aceptable, hay algo de overfitting")
else:
    print("   ❌ Overfitting significativo - considera:")
    print("      • Aumentar min_samples_leaf")
    print("      • Reducir max_depth")
    print("      • Aumentar ccp_alpha (regularización)")

print("\n💡 Próximos pasos:")
print("   1. Usar 'best_round2_model' para predecir en datos nuevos")
print("   2. Analizar feature importance del modelo")
print("   3. Visualizar errores para entender dónde falla")
print("   4. Si es necesario, hacer un grid search refinado alrededor")
print("      de los mejores hiperparámetros encontrados")

print("\n" + "="*60)
