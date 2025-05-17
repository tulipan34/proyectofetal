from flask import Flask, render_template, send_file, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import os
import io
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = '456'

# Cargar modelos
modelos = {
    'Regresión Logística': pickle.load(open('models/logistic_model.pkl', 'rb')),
    'Red Neuronal': pickle.load(open('models/ann_model.pkl', 'rb')),
    'SVM': pickle.load(open('models/svm_model.pkl', 'rb')),
    'Mapa Cognitivo Difuso': pickle.load(open('models/fcm_model.pkl', 'rb'))
}

# Columnas del formulario
columnas = [f'C{i}' for i in range(1, 31)]

# Campos legibles
campos_legibles = {
    'C1': 'Edad',
    'C2': 'IMC',
    'C3': 'Edad gestacional al parto',
    'C4': 'Gravidez',
    'C5': 'Paridad',
    'C6': 'Síntomas iniciales (0=edema, 1=hipertensión, 2=FGR)',
    'C7': 'Edad gestacional del inicio de síntomas',
    'C8': 'Días desde síntomas hasta parto',
    'C9': 'Edad gestacional del inicio de hipertensión',
    'C10': 'Días desde hipertensión hasta parto',
    'C11': 'Edad gestacional del inicio de edema',
    'C12': 'Días desde edema hasta parto',
    'C13': 'Edad gestacional del inicio de proteinuria',
    'C14': 'Días desde proteinuria hasta parto',
    'C15': 'Tratamiento expectante (0=No, 1=Sí)',
    'C16': 'Antihipertensivos antes de hospitalización (0=No, 1=Sí)',
    'C17': 'Antecedentes (0=No, 1=Hipertensión, 2=PCOS)',
    'C18': 'Presión sistólica máxima',
    'C19': 'Presión diastólica máxima',
    'C20': 'Razón del parto (0=HELLP, 1=Distress fetal, etc.)',
    'C21': 'Modo de parto (0=Cesárea, 1=Parto vaginal)',
    'C22': 'BNP máximo',
    'C23': 'Creatinina máxima',
    'C24': 'Ácido úrico máximo',
    'C25': 'Proteinuria máxima',
    'C26': 'Proteína total máxima',
    'C27': 'Albúmina máxima',
    'C28': 'ALT máxima',
    'C29': 'AST máxima',
    'C30': 'Plaquetas máximas',
    'Resultado': 'Resultado',
    'Modelo': 'Modelo',
    'Fecha': 'Fecha y Hora'
}

def predict_with_fcm(modelo_fcm, X_input):
    scaler = modelo_fcm['scaler']
    centers = modelo_fcm['centers']
    X_scaled = scaler.transform(X_input)
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
    y_pred = np.argmin(distances, axis=1)
    return y_pred

def hacer_prediccion(nombre_modelo, X):
    modelo = modelos[nombre_modelo]
    if nombre_modelo == 'Mapa Cognitivo Difuso':
        return predict_with_fcm(modelo, X)
    return modelo.predict(X)

@app.route('/')
def home():
    return render_template('index.html', columnas=columnas, campos_legibles=campos_legibles)

@app.route('/predecir_formulario', methods=['POST'])
def predict_manual():
    modelo = request.form.get('modelo')
    try:
        datos = {col: float(request.form[col]) for col in columnas}
    except ValueError:
        flash("⚠️ Todos los campos deben ser numéricos.", "danger")
        return redirect(url_for('home'))

    entrada_np = np.array(list(datos.values())).reshape(1, -1)
    try:
        y_pred = hacer_prediccion(modelo, entrada_np)
    except Exception as e:
        return f"Error al predecir: {e}"

    resultado = "FGR" if y_pred[0] == 1 else "Normal"

    # Guardar en historial
    datos_hist = datos.copy()
    datos_hist.update({
        'Modelo': modelo,
        'Resultado': resultado,
        'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Modo': 'Individual'
    })
    guardar_historial(datos_hist)

    return render_template('resultado.html',
                           resultado_manual=resultado,
                           modelo_seleccionado=modelo,
                           columnas=columnas,
                           request=request)

@app.route('/predecir_archivo', methods=['POST'])
def predict_batch():
    modelo = request.form['modelo']
    archivo = request.files['archivo']
    if not archivo or not archivo.filename.endswith('.xlsx'):
        return "Por favor sube un archivo .xlsx válido."

    df = pd.read_excel(archivo)
    requeridas = columnas + ['C31']
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        return f"Columnas faltantes: {', '.join(faltantes)}"

    X, y_true = df[columnas], df['C31']
    y_pred = hacer_prediccion(modelo, X)

    df['Resultado'] = ['FGR' if p == 1 else 'Normal' for p in y_pred]
    df['Modelo'] = modelo
    df['Fecha'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['Modo'] = 'Lote'
    guardar_historial(df)

    matriz = confusion_matrix(y_true, y_pred)
    exactitud = accuracy_score(y_true, y_pred)

    ruta_img = generar_matriz_confusion(matriz, modelo)

    return render_template('resultado_batch.html',
                           matriz=matriz.tolist(),
                           exactitud=round(exactitud * 100, 2),
                           modelo_seleccionado=modelo,
                           imagen_matriz=ruta_img)

def generar_matriz_confusion(matriz, modelo_nombre):
    fig, ax = plt.subplots()
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "FGR"], yticklabels=["Normal", "FGR"], ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"{modelo_nombre} - Matriz de Confusión")

    os.makedirs("static", exist_ok=True)
    nombre_img = f'static/confusion_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    plt.savefig(nombre_img, bbox_inches='tight')
    plt.close()
    return nombre_img

def guardar_historial(data):
    df_nuevo = pd.DataFrame([data]) if isinstance(data, dict) else data
    path = 'historial.csv'
    if os.path.exists(path):
        df_hist = pd.read_csv(path)
        df_hist = pd.concat([df_hist, df_nuevo], ignore_index=True)
    else:
        df_hist = df_nuevo
    df_hist.to_csv(path, index=False)

@app.route('/guardar_resultado', methods=['POST'])
def guardar_resultado():
    datos = {col: request.form.get(col) for col in columnas}
    datos.update({
        'Resultado': request.form.get('resultado'),
        'Modelo': request.form.get('modelo'),
        'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    df_original = pd.DataFrame([datos])
    df_legible = df_original.rename(columns=campos_legibles)

    vista_usuario = pd.DataFrame([
        [campos_legibles.get(k, k), v] for k, v in datos.items()
    ], columns=["Campo", "Valor Ingresado"])

    guardar_historial(df_original)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_original.to_excel(writer, index=False, sheet_name='Códigos Originales')
        df_legible.to_excel(writer, index=False, sheet_name='Datos Legibles')
        vista_usuario.to_excel(writer, index=False, sheet_name='Vista Usuario')
    output.seek(0)

    return send_file(output,
                     download_name='resultado_prediccion.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/descargar_historial')
def descargar_historial():
    return send_file('historial.csv', as_attachment=True)

@app.route('/historial')
def historial():
    registros = cargar_historial(ultimo=True)
    return render_template('historial.html', registros=registros)

@app.route('/historial_completo')
def historial_completo():
    registros = cargar_historial()
    return render_template('historial.html', registros=registros)

def cargar_historial(ultimo=False):
    if not os.path.exists('historial.csv'):
        return []
    df = pd.read_csv('historial.csv')
    return df.iloc[[-1]].to_dict(orient='records') if ultimo else df.to_dict(orient='records')

@app.route('/borrar_historial', methods=['POST'])
def borrar_historial():
    if os.path.exists('historial.csv'):
        os.remove('historial.csv')
        flash('✅ Historial borrado con éxito.', 'success')
    else:
        flash('⚠️ No hay historial para borrar.', 'warning')
    return redirect(url_for('historial'))

if __name__ == '__main__':
    app.run(debug=True)
