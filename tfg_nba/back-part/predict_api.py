# BACKEND (FastAPI) - serve teams from NBA API and predict with adjusted model keys
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import time
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, playergamelog
from fastapi.middleware.cors import CORSMiddleware

# Load models
modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")

app = FastAPI()

# Enable CORS for connecting with Deno frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
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


def obtener_caracteristicas_equipo_con_mvp(id_equipo, temporada='2024-25'):

    registro_juegos = teamgamelog.TeamGameLog(team_id=id_equipo, season=temporada, season_type_all_star='Regular Season')
    juegos = registro_juegos.get_data_frames()[0].head(5)

    estadisticas_equipo = []
    ids_mvp = []

    for id_juego in juegos['Game_ID']:

        caja = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=id_juego)
        df_jugadores = caja.get_data_frames()[0]
        df_equipos = caja.get_data_frames()[1]

        jugadores_equipo = df_jugadores[df_jugadores['TEAM_ID'] == id_equipo]
        if jugadores_equipo.empty:
            continue

        mvp = jugadores_equipo.sort_values(by='PTS', ascending=False).iloc[0]
        ids_mvp.append(mvp['PLAYER_ID'])

        fila_equipo = df_equipos[df_equipos['TEAM_ID'] == id_equipo]
        estadisticas_equipo.append(fila_equipo)
        time.sleep(1)
        

    if not estadisticas_equipo:
        raise ValueError("No se pudieron recuperar estadísticas del equipo")

    df_todos = pd.concat(estadisticas_equipo)
    perdidas_temp = df_todos['TO'].mean()
    
    estadisticas = {
        'teamScore_prom50': df_todos['PTS'].mean(),
        'assists_prom50': df_todos['AST'].mean(),
        'reboundsTotal_prom50': df_todos['REB'].mean(),
        'steals_prom50': df_todos['STL'].mean(),
        'blocks_prom50': df_todos['BLK'].mean(),
        'fieldGoalsPercentage_prom50': df_todos['FG_PCT'].mean(),
        'freeThrowsPercentage_prom50': df_todos['FT_PCT'].mean(),
        'threePointersPercentage_prom50': df_todos['FG3_PCT'].mean(),
        'plusMinusPoints_prom50': df_todos['PLUS_MINUS'].mean(),
        'eficiencia_ofensiva_prom50': df_todos['PTS'].mean() / (perdidas_temp + 1),
        'impacto_defensa_prom50': df_todos['STL'].mean() + df_todos['BLK'].mean(),
        'ratio_asistencias_turnovers_prom50': df_todos['AST'].mean() / (perdidas_temp + 1),
        'eficiencia_tiro_prom50': (df_todos['FG_PCT'].mean() + df_todos['FG3_PCT'].mean()) / 2,
        'home': 1
    }

    ids_mvp = list(set(ids_mvp))
    puntos_mvp, asistencias_mvp, rebotes_mvp, porcentaje_tiros_mvp = [], [], [], []

    for id_jugador in ids_mvp:

        registros = playergamelog.PlayerGameLog(player_id=id_jugador, season=temporada)
        df_mvp = registros.get_data_frames()[0].head(10)
        puntos_mvp.append(df_mvp['PTS'].mean())
        asistencias_mvp.append(df_mvp['AST'].mean())
        rebotes_mvp.append(df_mvp['REB'].mean())
        porcentaje_tiros_mvp.append(df_mvp['FG_PCT'].mean())
        time.sleep(1)


    estadisticas['MVP_PTS'] = np.mean(puntos_mvp) if puntos_mvp else 0
    estadisticas['MVP_AST'] = np.mean(asistencias_mvp) if asistencias_mvp else 0
    estadisticas['MVP_REB'] = np.mean(rebotes_mvp) if rebotes_mvp else 0
    estadisticas['MVP_FG_PCT'] = np.mean(porcentaje_tiros_mvp) if porcentaje_tiros_mvp else 0
    estadisticas['IMPACTO_MVP'] = (estadisticas['MVP_PTS'] * 1.0 + estadisticas['MVP_AST'] * 0.7 + estadisticas['MVP_REB'] * 0.5 + 10 * 0.3)

    return estadisticas

@app.get("/equipos")
def obtener_equipos():

    todos_equipos = nba_teams.get_teams()
    equipos = [{"id": t["id"], "name": t["full_name"]} for t in sorted(todos_equipos, key=lambda x: x["full_name"])]
    return equipos

@app.post("/predecir")
def predecir_desde_ids(entrada: EquiposInput):
    eq1 = obtener_caracteristicas_equipo_con_mvp(entrada.equipo1_id)
    eq2 = obtener_caracteristicas_equipo_con_mvp(entrada.equipo2_id)
    eq2['home'] = 0

        # Usar el orden exacto del modelo (igual que en predict-stats)
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
