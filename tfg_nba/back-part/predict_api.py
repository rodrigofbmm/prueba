from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
import time
import concurrent.futures

from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, playergamelog

# Cargar modelos
modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")

# Estadísticas cacheadas
with open("estadisticas_precalculadas.json") as f:
    cache = json.load(f)

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class EquiposInput(BaseModel):
    equipo1_id: int
    equipo2_id: int

class EstadisticasEquipos(BaseModel):
    equipo1: dict
    equipo2: dict

def obtener_datos_api(id_equipo, temporada="2024-25"):
    try:
        juegos = teamgamelog.TeamGameLog(team_id=id_equipo, season=temporada, season_type_all_star='Regular Season').get_data_frames()[0].head(10)
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
            time.sleep(0.5)

        if not estadisticas_equipo:
            raise ValueError("No hay datos")

        df = pd.concat(estadisticas_equipo)
        perdidas = df['TO'].mean()

        estadisticas = {
            'teamScore_prom50': df['PTS'].mean(),
            'assists_prom50': df['AST'].mean(),
            'reboundsTotal_prom50': df['REB'].mean(),
            'steals_prom50': df['STL'].mean(),
            'blocks_prom50': df['BLK'].mean(),
            'fieldGoalsPercentage_prom50': df['FG_PCT'].mean(),
            'freeThrowsPercentage_prom50': df['FT_PCT'].mean(),
            'threePointersPercentage_prom50': df['FG3_PCT'].mean(),
            'plusMinusPoints_prom50': df['PLUS_MINUS'].mean(),
            'eficiencia_ofensiva_prom50': df['PTS'].mean() / (perdidas + 1),
            'impacto_defensa_prom50': df['STL'].mean() + df['BLK'].mean(),
            'ratio_asistencias_turnovers_prom50': df['AST'].mean() / (perdidas + 1),
            'eficiencia_tiro_prom50': (df['FG_PCT'].mean() + df['FG3_PCT'].mean()) / 2,
            'home': 1
        }

        pts, ast, reb, fg = [], [], [], []
        for pid in set(ids_mvp[:3]):
            mvp_df = playergamelog.PlayerGameLog(player_id=pid, season=temporada).get_data_frames()[0].head(5)
            pts.append(mvp_df['PTS'].mean())
            ast.append(mvp_df['AST'].mean())
            reb.append(mvp_df['REB'].mean())
            fg.append(mvp_df['FG_PCT'].mean())
            time.sleep(0.5)

        estadisticas.update({
            'MVP_PTS': np.mean(pts) if pts else 0,
            'MVP_AST': np.mean(ast) if ast else 0,
            'MVP_REB': np.mean(reb) if reb else 0,
            'MVP_FG_PCT': np.mean(fg) if fg else 0
        })
        estadisticas['IMPACTO_MVP'] = (
            estadisticas['MVP_PTS'] +
            estadisticas['MVP_AST'] * 0.7 +
            estadisticas['MVP_REB'] * 0.5 +
            10 * 0.3
        )

        return estadisticas

    except Exception as e:
        raise e

def obtener_estadisticas_equipo(id_equipo):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(obtener_datos_api, id_equipo)
            return future.result(timeout=15)
    except:
        return cache.get(str(id_equipo), {})

@app.get("/equipos")
def equipos():
    equipos = nba_teams.get_teams()
    return [{"id": e["id"], "name": e["full_name"]} for e in sorted(equipos, key=lambda x: x["full_name"])]

@app.post("/predecir")
def predecir_ids(datos: EquiposInput):
    eq1 = obtener_estadisticas_equipo(datos.equipo1_id)
    eq2 = obtener_estadisticas_equipo(datos.equipo2_id)
    eq2["home"] = 0

    claves = [
        'teamScore_prom50', 'assists_prom50', 'blocks_prom50', 'steals_prom50',
        'fieldGoalsPercentage_prom50', 'threePointersPercentage_prom50', 'freeThrowsPercentage_prom50',
        'reboundsTotal_prom50', 'plusMinusPoints_prom50', 'eficiencia_ofensiva_prom50',
        'impacto_defensa_prom50', 'ratio_asistencias_turnovers_prom50', 'eficiencia_tiro_prom50',
        'MVP_PTS', 'MVP_AST', 'MVP_REB', 'MVP_FG_PCT', 'IMPACTO_MVP', 'home'
    ]

    dif = np.array([eq1.get(k, 0) - eq2.get(k, 0) for k in claves])
    X = pd.DataFrame([dif], columns=[f'entreno_{k}' for k in claves])
    X_proc = escalador.transform(imputador.transform(X))
    prob = modelo.predict_proba(X_proc)[0][1]

    return {"probabilidad_victoria_local": round(prob * 100, 2)}

@app.post("/predecir-estadisticas")
def predecir_estadisticas(datos: EstadisticasEquipos):
    mapeo = {
        'teamScore': 'teamScore_prom50', 'assists': 'assists_prom50', 'blocks': 'blocks_prom50',
        'steals': 'steals_prom50', 'fieldGoalsPercentage': 'fieldGoalsPercentage_prom50',
        'threePointersPercentage': 'threePointersPercentage_prom50', 'freeThrowsPercentage': 'freeThrowsPercentage_prom50',
        'reboundsTotal': 'reboundsTotal_prom50', 'plusMinusPoints': 'plusMinusPoints_prom50',
        'eficienciaOfensiva': 'eficiencia_ofensiva_prom50', 'impactoDefensa': 'impacto_defensa_prom50',
        'ratioAsistenciasTurnovers': 'ratio_asistencias_turnovers_prom50', 'eficienciaTiro': 'eficiencia_tiro_prom50',
        'MVP_points': 'MVP_PTS', 'MVP_assists': 'MVP_AST', 'MVP_rebounds': 'MVP_REB',
        'MVP_fg': 'MVP_FG_PCT', 'impactoMVP': 'IMPACTO_MVP', 'isHome': 'home'
    }

    claves = list(mapeo.values())
    eq1 = {mapeo[k]: v for k, v in datos.equipo1.items() if k in mapeo}
    eq2 = {mapeo[k]: v for k, v in datos.equipo2.items() if k in mapeo}

    dif = np.array([eq1.get(k, 0) - eq2.get(k, 0) for k in claves])
    X = pd.DataFrame([dif], columns=[f'entreno_{k}' for k in claves])
    X_proc = escalador.transform(imputador.transform(X))
    prob = modelo.predict_proba(X_proc)[0][1]

    return {"probabilidad_victoria_local": round(prob * 100, 2)}

