from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelos
modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")

# Cargar estadísticas precalculadas
with open("estadisticas_precalculadas.json") as f:
    estadisticas_cacheadas = json.load(f)

app = FastAPI()

# CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EquiposInput(BaseModel):
    equipo1_id: int
    equipo2_id: int

class EstadisticasEquipos(BaseModel):
    equipo1: dict
    equipo2: dict

@app.get("/equipos")
def obtener_equipos():
    # Puedes reemplazar esto por una lista fija si prefieres
    from nba_api.stats.static import teams as nba_teams
    todos_equipos = nba_teams.get_teams()
    equipos = [{"id": t["id"], "name": t["full_name"]} for t in sorted(todos_equipos, key=lambda x: x["full_name"])]
    return equipos

@app.post("/predecir")
def predecir_desde_ids(entrada: EquiposInput):
    eq1 = estadisticas_cacheadas[str(entrada.equipo1_id)]
    eq2 = estadisticas_cacheadas[str(entrada.equipo2_id)]
    eq2["home"] = 0  # visitante

    claves_modelo = [
        'teamScore_prom50', 'assists_prom50', 'blocks_prom50', 'steals_prom50',
        'fieldGoalsPercentage_prom50', 'threePointersPercentage_prom50', 'freeThrowsPercentage_prom50',
        'reboundsTotal_prom50', 'plusMinusPoints_prom50',
        'eficiencia_ofensiva_prom50', 'impacto_defensa_prom50', 'ratio_asistencias_turnovers_prom50',
        'eficiencia_tiro_prom50',
        'MVP_PTS', 'MVP_AST', 'MVP_REB', 'MVP_FG_PCT',
        'IMPACTO_MVP', 'home'
    ]

    v1 = [eq1.get(k, 0) for k in claves_modelo]
    v2 = [eq2.get(k, 0) for k in claves_modelo]
    diferencia = np.array(v1) - np.array(v2)

    X = pd.DataFrame([diferencia], columns=[f'entreno_{k}' for k in claves_modelo])
    X_imputado = imputador.transform(X)
    X_escalado = escalador.transform(X_imputado)
    probabilidad = modelo.predict_proba(X_escalado)[0][1]

    return {"probabilidad_victoria_local": round(probabilidad * 100, 2)}

@app.post("/predecir-estadisticas")
def predecir_con_estadisticas(datos: EstadisticasEquipos):
    frontend_a_modelo = {
        'teamScore': 'teamScore_prom50',
        'assists': 'assists_prom50',
        'blocks': 'blocks_prom50',
        'steals': 'steals_prom50',
        'fieldGoalsPercentage': 'fieldGoalsPercentage_prom50',
        'threePointersPercentage': 'threePointersPercentage_prom50',
        'freeThrowsPercentage': 'freeThrowsPercentage_prom50',
        'reboundsTotal': 'reboundsTotal_prom50',
        'plusMinusPoints': 'plusMinusPoints_prom50',
        'eficienciaOfensiva': 'eficiencia_ofensiva_prom50',
        'impactoDefensa': 'impacto_defensa_prom50',
        'ratioAsistenciasTurnovers': 'ratio_asistencias_turnovers_prom50',
        'eficienciaTiro': 'eficiencia_tiro_prom50',
        'MVP_points': 'MVP_PTS',
        'MVP_assists': 'MVP_AST',
        'MVP_rebounds': 'MVP_REB',
        'MVP_fg': 'MVP_FG_PCT',
        'impactoMVP': 'IMPACTO_MVP',
        'isHome': 'home'
    }

    claves_modelo = list(frontend_a_modelo.values())

    datos_procesados = {}
    for clave_equipo, datos_equipo in [("equipo1", datos.equipo1), ("equipo2", datos.equipo2)]:
        datos_procesados[clave_equipo] = {}
        for clave_frontend, valor in datos_equipo.items():
            if clave_frontend in frontend_a_modelo:
                clave_modelo = frontend_a_modelo[clave_frontend]
                datos_procesados[clave_equipo][clave_modelo] = valor

    v1 = [datos_procesados["equipo1"].get(k, 0) for k in claves_modelo]
    v2 = [datos_procesados["equipo2"].get(k, 0) for k in claves_modelo]
    diferencia = np.array(v1) - np.array(v2)

    X = pd.DataFrame([diferencia], columns=[f'entreno_{k}' for k in claves_modelo])
    X_imputado = imputador.transform(X)
    X_escalado = escalador.transform(X_imputado)
    probabilidad = modelo.predict_proba(X_escalado)[0][1]

    return {"probabilidad_victoria_local": round(probabilidad * 100, 2)}

