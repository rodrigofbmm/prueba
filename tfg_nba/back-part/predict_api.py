from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
import concurrent.futures
import time
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, playergamelog

# Cargar modelos
modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")

# Cargar estadísticas precalculadas como fallback
with open("estadisticas_precalculadas.json") as f:
    estadisticas_fallback = json.load(f)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class EquiposInput(BaseModel):
    equipo1_id: int
    equipo2_id: int

class EstadisticasEquipos(BaseModel):
    equipo1: dict
    equipo2: dict

def obtener_datos_nba_api(id_equipo, temporada="2024-25"):
    """Intenta obtener datos de la NBA API con timeout de 15 segundos"""
    try:
        print(f"🔄 Conectando con NBA API para equipo {id_equipo}")
        
        # Obtener últimos 10 juegos
        registro_juegos = teamgamelog.TeamGameLog(
            team_id=id_equipo, 
            season=temporada, 
            season_type_all_star='Regular Season',
            timeout=8
        )
        juegos = registro_juegos.get_data_frames()[0].head(10)

        estadisticas_equipo = []
        ids_mvp = []

        for idx, id_juego in enumerate(juegos['Game_ID']):
            try:
                caja = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=id_juego, timeout=5
                )
                df_jugadores = caja.get_data_frames()[0]
                df_equipos = caja.get_data_frames()[1]

                jugadores_equipo = df_jugadores[df_jugadores['TEAM_ID'] == id_equipo]
                if not jugadores_equipo.empty:
                    mvp = jugadores_equipo.sort_values(by='PTS', ascending=False).iloc[0]
                    ids_mvp.append(mvp['PLAYER_ID'])

                fila_equipo = df_equipos[df_equipos['TEAM_ID'] == id_equipo]
                estadisticas_equipo.append(fila_equipo)
                time.sleep(0.5)  # Pausa entre peticiones
                    
            except Exception as e:
                print(f"  ⚠️ Error en juego {id_juego}: {e}")
                continue

        if not estadisticas_equipo:
            raise ValueError("Sin datos válidos de juegos")

        # Procesar estadísticas del equipo
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

        # Procesar estadísticas MVP
        puntos, asistencias, rebotes, fg_pct = [], [], [], []
        for pid in set(ids_mvp[:3]):  # Solo los 3 primeros para ahorrar tiempo
            try:
                df_mvp = playergamelog.PlayerGameLog(
                    player_id=pid, season=temporada, timeout=5
                ).get_data_frames()[0].head(5)
                puntos.append(df_mvp['PTS'].mean())
                asistencias.append(df_mvp['AST'].mean())
                rebotes.append(df_mvp['REB'].mean())
                fg_pct.append(df_mvp['FG_PCT'].mean())
                time.sleep(0.5)
            except:
                continue

        estadisticas.update({
            'MVP_PTS': np.mean(puntos) if puntos else 20.0,
            'MVP_AST': np.mean(asistencias) if asistencias else 6.0,
            'MVP_REB': np.mean(rebotes) if rebotes else 8.0,
            'MVP_FG_PCT': np.mean(fg_pct) if fg_pct else 0.45,
        })
        
        estadisticas['IMPACTO_MVP'] = (
            estadisticas['MVP_PTS'] * 1.0 +
            estadisticas['MVP_AST'] * 0.7 +
            estadisticas['MVP_REB'] * 0.5 +
            10 * 0.3
        )

        print(f"✅ Datos NBA API obtenidos para equipo {id_equipo}")
        return estadisticas

    except Exception as e:
        print(f"❌ Error NBA API para equipo {id_equipo}: {e}")
        raise e

def obtener_estadisticas_equipo(id_equipo):
    """
    1. Intenta NBA API con timeout de 15 segundos
    2. Si falla, usa datos del JSON
    """
    try:
        # Intentar API con timeout de 15 segundos
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(obtener_datos_nba_api, id_equipo)
            resultado = future.result(timeout=15)
            return resultado
            
    except concurrent.futures.TimeoutError:
        print(f"⏰ Timeout NBA API para equipo {id_equipo}, usando fallback")
    except Exception as e:
        print(f"⚠️ Error NBA API para equipo {id_equipo}: {e}, usando fallback")
    
    # Fallback: usar datos del JSON
    if str(id_equipo) in estadisticas_fallback:
        print(f"📦 Usando datos fallback para equipo {id_equipo}")
        return estadisticas_fallback[str(id_equipo)]
    else:
        print(f"⚠️ Sin datos para equipo {id_equipo}, usando valores por defecto")
        return {
            'teamScore_prom50': 110.0, 'assists_prom50': 25.0, 'reboundsTotal_prom50': 45.0,
            'steals_prom50': 8.0, 'blocks_prom50': 5.0, 'fieldGoalsPercentage_prom50': 0.45,
            'freeThrowsPercentage_prom50': 0.75, 'threePointersPercentage_prom50': 0.35,
            'plusMinusPoints_prom50': 0.0, 'eficiencia_ofensiva_prom50': 8.0,
            'impacto_defensa_prom50': 13.0, 'ratio_asistencias_turnovers_prom50': 2.0,
            'eficiencia_tiro_prom50': 0.40, 'MVP_PTS': 20.0, 'MVP_AST': 6.0,
            'MVP_REB': 8.0, 'MVP_FG_PCT': 0.45, 'IMPACTO_MVP': 35.0, 'home': 1
        }

@app.get("/equipos")
def obtener_equipos():
    """Obtener lista de equipos NBA"""
    try:
        equipos = nba_teams.get_teams()
        equipos_ordenados = sorted(equipos, key=lambda x: x["full_name"])
        return [{"id": e["id"], "name": e["full_name"]} for e in equipos_ordenados]
    except Exception as e:
        print(f"Error obteniendo equipos: {e}")
        return [
            {"id": 1610612738, "name": "Boston Celtics"},
            {"id": 1610612751, "name": "Brooklyn Nets"},
            {"id": 1610612752, "name": "New York Knicks"},
            {"id": 1610612747, "name": "Los Angeles Lakers"},
            {"id": 1610612744, "name": "Golden State Warriors"}
        ]

@app.post("/predecir")
def predecir_desde_ids(entrada: EquiposInput):
    """Predicción desde IDs de equipos"""
    try:
        print(f"🏀 Predicción: {entrada.equipo1_id} vs {entrada.equipo2_id}")
        
        # Obtener estadísticas (API primero, JSON como fallback)
        eq1 = obtener_estadisticas_equipo(entrada.equipo1_id)
        eq2 = obtener_estadisticas_equipo(entrada.equipo2_id)
        eq2["home"] = 0  # Equipo visitante

        # Claves para el modelo
        claves = [
            'teamScore_prom50', 'assists_prom50', 'blocks_prom50', 'steals_prom50',
            'fieldGoalsPercentage_prom50', 'threePointersPercentage_prom50', 'freeThrowsPercentage_prom50',
            'reboundsTotal_prom50', 'plusMinusPoints_prom50', 'eficiencia_ofensiva_prom50', 
            'impacto_defensa_prom50', 'ratio_asistencias_turnovers_prom50', 'eficiencia_tiro_prom50',
            'MVP_PTS', 'MVP_AST', 'MVP_REB', 'MVP_FG_PCT', 'IMPACTO_MVP', 'home'
        ]

        # Calcular diferencias
        v1 = [eq1.get(k, 0) for k in claves]
        v2 = [eq2.get(k, 0) for k in claves]
        diferencia = np.array(v1) - np.array(v2)

        # Predicción
        X = pd.DataFrame([diferencia], columns=[f'entreno_{k}' for k in claves])
        X_imputado = imputador.transform(X)
        X_escalado = escalador.transform(X_imputado)
        probabilidad = modelo.predict_proba(X_escalado)[0][1]

        resultado = {"probabilidad_victoria_local": round(probabilidad * 100, 2)}
        print(f"✅ Predicción completada: {resultado}")
        return resultado
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return {"error": "Error en la predicción", "probabilidad_victoria_local": 50.0}

@app.post("/predecir-estadisticas")
def predecir_con_estadisticas(datos: EstadisticasEquipos):
    """Predicción desde estadísticas manuales"""
    try:
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
        d1 = {mapeo[k]: v for k, v in datos.equipo1.items() if k in mapeo}
        d2 = {mapeo[k]: v for k, v in datos.equipo2.items() if k in mapeo}

        v1 = [d1.get(k, 0) for k in claves]
        v2 = [d2.get(k, 0) for k in claves]
        diferencia = np.array(v1) - np.array(v2)

        X = pd.DataFrame([diferencia], columns=[f'entreno_{k}' for k in claves])
        X_imputado = imputador.transform(X)
        X_escalado = escalador.transform(X_imputado)
        probabilidad = modelo.predict_proba(X_escalado)[0][1]

        return {"probabilidad_victoria_local": round(probabilidad * 100, 2)}
        
    except Exception as e:
        print(f"❌ Error en predicción manual: {e}")
        return {"error": "Error en la predicción", "probabilidad_victoria_local": 50.0}
