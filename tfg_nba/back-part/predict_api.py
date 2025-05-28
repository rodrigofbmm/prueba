from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, json, numpy as np, pandas as pd, concurrent.futures, time
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, playergamelog

# Cargar modelos y datos
modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")

try:
    with open("estadisticas_precalculadas.json") as f:
        cache = json.load(f)
except:
    cache = {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class EquiposInput(BaseModel): 
    equipo1_id: int
    equipo2_id: int

class EstadisticasEquipos(BaseModel): 
    equipo1: dict
    equipo2: dict

def obtener_datos_api(id_equipo):
    try:
        # Obtener juegos del equipo
        juegos = teamgamelog.TeamGameLog(team_id=id_equipo, season="2024-25", timeout=8).get_data_frames()[0].head(10)
        estadisticas_equipo = []
        ids_mvp = []
        
        # Procesar cada juego
        for id_juego in juegos['Game_ID']:
            try:
                caja = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=id_juego, timeout=5)
                df_jugadores = caja.get_data_frames()[0]
                df_equipos = caja.get_data_frames()[1]
                
                # Estadísticas del equipo
                fila_equipo = df_equipos[df_equipos['TEAM_ID'] == id_equipo]
                if not fila_equipo.empty:
                    estadisticas_equipo.append(fila_equipo)
                    
                    # MVP del juego
                    jugadores_equipo = df_jugadores[df_jugadores['TEAM_ID'] == id_equipo]
                    if not jugadores_equipo.empty:
                        mvp = jugadores_equipo.sort_values('PTS', ascending=False).iloc[0]
                        ids_mvp.append(mvp['PLAYER_ID'])
                
                time.sleep(0.5)
            except:
                continue
                
        if not estadisticas_equipo:
            raise ValueError("Sin datos")
            
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
        
        # Procesar estadísticas MVP (solo los primeros 3 para ahorrar tiempo)
        puntos, asistencias, rebotes, fg_pct = [], [], [], []
        for pid in list(set(ids_mvp))[:3]:
            try:
                df_mvp = playergamelog.PlayerGameLog(player_id=pid, season="2024-25", timeout=5).get_data_frames()[0].head(5)
                puntos.append(df_mvp['PTS'].mean())
                asistencias.append(df_mvp['AST'].mean())
                rebotes.append(df_mvp['REB'].mean())
                fg_pct.append(df_mvp['FG_PCT'].mean())
                time.sleep(0.5)
            except:
                continue
        
        # Estadísticas MVP con valores por defecto si no hay datos
        estadisticas.update({
            'MVP_PTS': np.mean(puntos) if puntos else 0,
            'MVP_AST': np.mean(asistencias) if asistencias else 0,
            'MVP_REB': np.mean(rebotes) if rebotes else 0,
            'MVP_FG_PCT': np.mean(fg_pct) if fg_pct else 0,
        })
        
        # Calcular impacto MVP
        estadisticas['IMPACTO_MVP'] = (
            estadisticas['MVP_PTS'] * 1.0 +
            estadisticas['MVP_AST'] * 0.7 +
            estadisticas['MVP_REB'] * 0.5 +
            10 * 0.3
        )
        
        return estadisticas
    except:
        raise

def obtener_estadisticas_equipo(id_equipo):
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            print(f"Obteniendo datos desde la API para equipo {id_equipo}")
            return ex.submit(obtener_datos_api, id_equipo).result(timeout=15)
    except:
        print(f"Usando cache para equipo {id_equipo}")
        return cache.get(str(id_equipo), {})

@app.get("/equipos")
def equipos():
    try:
        equipos = nba_teams.get_teams()
        return [{"id": e["id"], "name": e["full_name"]} for e in sorted(equipos, key=lambda x: x["full_name"])]
    except:
        return []

@app.post("/predecir")
def predecir_ids(entrada: EquiposInput):
    try:
        eq1 = obtener_estadisticas_equipo(entrada.equipo1_id)
        eq2 = obtener_estadisticas_equipo(entrada.equipo2_id)
        
        if not eq1 or not eq2:
            return {"error": "No se pudieron obtener estadísticas"}
            
        eq2["home"] = 0  # Visitante
        
        # ORDEN EXACTO de características que espera el modelo
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
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/predecir-estadisticas")  
def predecir_estadisticas(datos: EstadisticasEquipos):
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
        
        eq1 = {mapeo[k]: v for k, v in datos.equipo1.items() if k in mapeo}
        eq2 = {mapeo[k]: v for k, v in datos.equipo2.items() if k in mapeo}
        
        # MISMO ORDEN que en /predecir
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
        
    except Exception as e:
        return {"error": str(e)}
