import pyodbc
import pandas as pd
import numpy as np

# WARNING SI ES NECESARIO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# CONEXIÓN A LA BASE DE DATOS 
DATABASE = "PROJECT_OPT_NUM"
SERVER = r"localhost\SQLEXPRESS"      
connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=PROJECT_OPT_NUM;"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
)
# Crear conexión
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()  
print("Conexión a SQL Server: OK")

df = pd.read_sql("SELECT * FROM supply_chain_regression;", conn)
print("Filas, columnas:", df.shape)
print(df.head())

cursor.close()
conn.close()

# VARIABLES DEL MODELO 
# Variable de respuesta
y = df["Price"].values              # precio del producto cosmético

# Covariables
X = df[[
    "manufacturing_costs",          # costos de producción
    "shipping_costs",               # costos de envío    
    "production_volumes",           # volúmenes de producción 
    "manufacturing_lead_time"
]].values

# ESCALAMIENTO DE LAS VARIABLES
# Covariables
X_mean = X.mean(axis=0) # media por columna
X_std  = X.std(axis=0) # desviación estándar por columna
X_scaled = (X - X_mean) / X_std

# Variable de respuesta
y_mean = y.mean()
y_std  = y.std()
y_scaled = (y - y_mean) / y_std

# Usaremos las variables escaladas y agregaremos intercepto
n = X_scaled.shape[0]
X_used = np.column_stack([np.ones(n), X_scaled])   # intercepto + 4 covariables => 5 betas
y_used = y_scaled

# FUNCIÓN OBJETIVO
def FuncionObjetivo(beta):
    n = len(y_used)
    residuo = X_used @ beta - y_used
    return (1/(2*n)) * np.dot(residuo, residuo)

# GRADIENTE
def gradiente(beta):
    n = len(y_used)
    return (1/n) * X_used.T @ (X_used @ beta - y_used)

# HESSIANO
def hessiano(beta):
    n = len(y_used)
    return (1/n) * X_used.T @ X_used

# AUTOGRAD 
import autograd.numpy as anp
from autograd import grad as ag_grad
from autograd import hessian as ag_hessian
    
# Versiones autograd de X y Y (constantes) para poder derivar en beta
X_ag = anp.array(X_used)
y_ag = anp.array(y_used)

def FuncionObjetivo_ag(beta):
    n = y_ag.shape[0]
    r = anp.dot(X_ag, beta) - y_ag
    return (1/(2*n)) * anp.dot(r, r)
grad_ag = ag_grad(FuncionObjetivo_ag)
hess_ag = ag_hessian(FuncionObjetivo_ag)

def verificar_derivadas(beta_test):
    g_analitico = gradiente(beta_test)
    H_analitico = hessiano(beta_test)
    g_auto = np.array(grad_ag(beta_test))
    H_auto = np.array(hess_ag(beta_test))
    print("\n--- Verificación Autograd ---")
    print("||g_analitico - g_auto|| =", np.linalg.norm(g_analitico - g_auto))
    print("||H_analitico - H_auto||_F =", np.linalg.norm(H_analitico - H_auto, ord='fro'))

# GOLDEN SEARCH
def f_alpha(beta, d, alpha, FuncionObjetivo):
    return FuncionObjetivo(beta + alpha * d)

def golden_search(beta, d, a, b, epsilon, FuncionObjetivo, GR=0.61803398875, MaxIter=50):
    if a > b:
        a, b = b, a
    dist = GR * (b - a)
    lambda_1 = b - dist
    lambda_2 = a + dist
    i = 0
    while abs(lambda_1 - lambda_2) > epsilon and i <= MaxIter:
        f1 = f_alpha(beta, d, lambda_1, FuncionObjetivo)
        f2 = f_alpha(beta, d, lambda_2, FuncionObjetivo)
        if f1 > f2:
            a = lambda_1
        elif f1 < f2:
            b = lambda_2
        else:
            a = lambda_1
            b = lambda_2
        dist = GR * (b - a)
        lambda_1 = b - dist
        lambda_2 = a + dist
        i += 1
    return (lambda_1 + lambda_2) / 2.0

# MÉTODO DE NEWTON
def newton_method(x0, max_it=50, tol=1e-6, use_golden=False, a=0.0, b=1.0, eps_gs=1e-5):
    xk = x0.copy()
    history = []
    k = 0
    norm_g = np.linalg.norm(np.array(grad_ag(xk)))
    while k < max_it and norm_g >= tol:
        gk = np.array(grad_ag(xk))
        Hk = np.array(hess_ag(xk))
        norm_g = np.linalg.norm(gk)
        fk = FuncionObjetivo_ag(xk)
        history.append((k, fk, norm_g, xk.copy()))
        dk = -np.linalg.solve(Hk, gk)
        alpha_step = golden_search(xk, dk, a, b, eps_gs, FuncionObjetivo_ag) if use_golden else 1.0
        xk = xk + alpha_step * dk
        k += 1
    return xk, history

# TRUST REGION
def trust_region(x0, alpha=1.0, max_it=50, tol=1e-6, use_golden=False, a=0.0, b=1.0, eps_gs=1e-5):
    xk = x0.copy()
    history = []
    k = 0
    norm_g = np.linalg.norm(gradiente(xk))
    while k < max_it and norm_g >= tol:
        gk = gradiente(xk)
        Hk = hessiano(xk)
        norm_g = np.linalg.norm(gk)
        fk = FuncionObjetivo(xk)
        history.append((k, fk, norm_g, xk.copy()))
        dk = -np.linalg.solve(Hk + alpha*np.eye(len(xk)), gk)
        alpha_step = golden_search(xk, dk, a, b, eps_gs, FuncionObjetivo) if use_golden else 1.0
        xk = xk + alpha_step * dk
        k += 1
    return xk, history

# DESCENSO DEL GRADIENTE 
def gradient_descent(gradiente_func, x0, learn_rate=0.1, max_it=2000, tol=1e-6):
    xk = x0.copy()
    history = []
    for k in range(max_it):
        gk = gradiente_func(xk)
        norm_g = np.linalg.norm(gk)
        fk = FuncionObjetivo(xk)
        history.append((k, fk, norm_g, xk.copy()))
        if norm_g < tol:
            break
        xk = xk - learn_rate * gk
    return xk, history

# EJECUCIÓN Y TABLA DE RESULTADOS
beta0 = np.zeros(X_used.shape[1])  # 5 parámetros: intercepto + 4 betas
verificar_derivadas(beta0)
beta_newton, hist_newton = newton_method(beta0, max_it=50, tol=1e-6, use_golden=True)
beta_tr, hist_tr = trust_region(beta0, alpha=1.0, max_it=50, tol=1e-6, use_golden=True) 
beta_gd, hist_gd = gradient_descent(gradiente, beta0, learn_rate=0.1)

print("\n--- Resultados finales de Newton con Autograd---")
# Método Newton
print("Función objetivo f(beta_newton):")
print(FuncionObjetivo(beta_newton))
print("\nGradiente ∇f(beta_newton):")
print(gradiente(beta_newton))
print("\nHessiano H(beta_newton):")
print(hessiano(beta_newton))
print("\nBeta (Newton):", beta_newton)
print("\nIteraciones Newton:", len(hist_newton))

# Método Trust Region
print("\n--- Resultados finales de Trust Region ---")
print("Función objetivo f(beta_tr):")
print(FuncionObjetivo(beta_tr))
print("\nGradiente ∇f(beta_tr):")
print(gradiente(beta_tr))
print("\nHessiano H(beta_tr):")
print(hessiano(beta_tr))
print("\nBeta (Trust Region):", beta_tr)
print("\nIteraciones Trust Region:", len(hist_tr))

# Método Descenso del Gradiente
print("\n--- Resultados Descenso del Gradiente ---")
print("Función objetivo f(beta_gd):")
print(FuncionObjetivo(beta_gd))
print("\nNorma del gradiente final:", np.linalg.norm(gradiente(beta_gd)))
print("\nBeta (GD):", beta_gd)
print("\nIteraciones GD:", len(hist_gd))

# MODELO DE REGRESIÓN LINEAL
def modelo_regresion_lineal(beta, nombres_variables):
    print(f"y_hat = {beta[0]:.6f}")
    for i in range(1, len(beta)):
        signo = "+" if beta[i] >= 0 else "-"
        print(f"        {signo} {abs(beta[i]):.6f} * {nombres_variables[i-1]}")
nombres_vars = [
    "manufacturing_costs_scaled",
    "shipping_costs_scaled",
    "production_volumes_scaled",
    "manufacturing_lead_time_scaled"
]
# 1. Método de Newton
print("\n --- Modelo de regresión lineal (método Newton) ---")
modelo_regresion_lineal(beta_newton, nombres_vars)
df_newton = pd.DataFrame([(k, f, ng) for (k, f, ng, _) in hist_newton],
                         columns=["iter", "f(beta)", "||grad||"])
# Tabla de iteraciones (Newton)
print("\nTabla Newton (primeras 10):")
print(df_newton.head(10))

# 2. Método Trust Region
print("\n --- Modelo de regresión lineal (método trust region) ---")
modelo_regresion_lineal(beta_tr, nombres_vars)
df_tr = pd.DataFrame([(k, f, ng) for (k, f, ng, _) in hist_tr],
                     columns=["iter", "f(beta)", "||grad||"])
# Tabla de iteraciones (Trust Region)
print("\nTabla Trust Region (primeras 10):")
print(df_tr.head(10))

# MODELO NO LINEAL: POTENCIA
# y = θ0 * x1^θ1 * x2^θ2 * x3^θ3 * x4^θ4

# Covariables SIN estandarizar
X_nl = df[[
    "manufacturing_costs",
    "shipping_costs",
    "production_volumes",
    "manufacturing_lead_time"
]].values

# Variable respuesta
y_nl = y_used.reshape(-1)

# Evitar ceros
eps = 1e-6
X_nl = X_nl + eps
n = y_nl.shape[0]

# Predicción no lineal
def y_hat_nl(theta):
    return (
        theta[0]
        * (X_nl[:, 0] ** theta[1])
        * (X_nl[:, 1] ** theta[2])
        * (X_nl[:, 2] ** theta[3])
        * (X_nl[:, 3] ** theta[4])
    )

# Función objetivo
def FuncionObjetivo_nl(theta):
    r = y_hat_nl(theta) - y_nl
    return (1 / (2 * n)) * anp.dot(r, r)

# Gradiente y Hessiano (Autograd)
grad_nl = ag_grad(FuncionObjetivo_nl)
hess_nl = ag_hessian(FuncionObjetivo_nl)

# Punto inicial
theta0 = anp.array([1.0, 0.1, 0.1, 0.1, 0.1])

# Parámetros:
max_it = 50          # máximo de iteraciones
e = 1e-6             # tolerancia del gradiente
alpha_trust = 1.0   # parámetro trust region

# 1. Método de Newton 
theta_newton_nl = theta0.copy()
k = 0
while k < max_it and np.linalg.norm(np.array(grad_nl(theta_newton_nl))) >= e:
    g = np.array(grad_nl(theta_newton_nl))
    H = np.array(hess_nl(theta_newton_nl))
    d = np.linalg.solve(H, -g)
    alpha = golden_search(theta_newton_nl, d, 0.0, 1.0, 1e-5, FuncionObjetivo_nl)
    theta_newton_nl = theta_newton_nl + alpha * d
    k += 1

# 2. Método Trust Region 
theta_trust_nl = theta0.copy()
k = 0
while k < max_it and np.linalg.norm(np.array(grad_nl(theta_trust_nl))) >= e:
    g = np.array(grad_nl(theta_trust_nl))
    H = np.array(hess_nl(theta_trust_nl))
    d = np.linalg.solve(H + alpha_trust * np.eye(len(theta_trust_nl)), -g)
    theta_trust_nl = theta_trust_nl + d
    k += 1

# Ejecución
print("\n --- MODELO NO LINEAL (Potencia) ---")
print("\n Método de Newton:")
print("θ0 =", theta_newton_nl[0])
print("θ1 =", theta_newton_nl[1])
print("θ2 =", theta_newton_nl[2])
print("θ3 =", theta_newton_nl[3])
print("θ4 =", theta_newton_nl[4])

print("\nTrust Region:")
print("θ0 =", theta_trust_nl[0])
print("θ1 =", theta_trust_nl[1])
print("θ2 =", theta_trust_nl[2])
print("θ3 =", theta_trust_nl[3])
print("θ4 =", theta_trust_nl[4])

df_results = pd.DataFrame([
    ["Newton", *beta_newton, FuncionObjetivo(beta_newton), len(hist_newton)],
    ["Trust Region", *beta_tr, FuncionObjetivo(beta_tr), len(hist_tr)],
    ["Gradient Descent", *beta_gd, FuncionObjetivo(beta_gd), len(hist_gd)]
], columns=[
    "metodo", "beta0", "beta1", "beta2", "beta3", "beta4",
    "f_beta", "iteraciones"
])

conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
for _, row in df_results.iterrows():
    cursor.execute("""
        INSERT INTO resultados_modelo
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(row))
conn.commit()
cursor.close()
conn.close()