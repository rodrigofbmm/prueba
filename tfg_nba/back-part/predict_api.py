# BACKEND (FastAPI) - serve teams from NBA API and predict with adjusted model keys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import time
import logging
import traceback
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, playergamelog
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load models with error handling
try:
    modelo = joblib.load("../modelo-definitivo/mejor_modelo.pkl")
    escalador = joblib.load("../modelo-definitivo/escalado_equipo.pkl")
    imputador = joblib.load("../modelo-definitivo/imputo_equipo.pkl")
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    raise

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
    logger.info(f"🏀 Starting data fetch for team ID: {id_equipo}")
    
    try:
        # Step 1: Get team games
        logger.info(f"📊 Fetching game log for team {id_equipo}...")
        registro_juegos = teamgamelog.TeamGameLog(
            team_id=id_equipo, 
            season=temporada, 
            season_type_all_star='Regular Season'
        )
        juegos = registro_juegos.get_data_frames()[0].head(2)
        logger.info(f"✅ Found {len(juegos)} games for team {id_equipo}")
        
        if juegos.empty:
            logger.error(f"❌ No games found for team {id_equipo}")
            raise ValueError(f"No games found for team {id_equipo}")

        estadisticas_equipo = []
        ids_mvp = []

        # Step 2: Process each game
        for idx, id_juego in enumerate(juegos['Game_ID']):
            logger.info(f"🎮 Processing game {idx+1}/{len(juegos)}: {id_juego}")
            
            try:
                caja = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=id_juego)
                df_jugadores = caja.get_data_frames()[0]
                df_equipos = caja.get_data_frames()[1]

                jugadores_equipo = df_jugadores[df_jugadores['TEAM_ID'] == id_equipo]
                if jugadores_equipo.empty:
                    logger.warning(f"⚠️ No players found for team {id_equipo} in game {id_juego}")
                    continue

                mvp = jugadores_equipo.sort_values(by='PTS', ascending=False).iloc[0]
                ids_mvp.append(mvp['PLAYER_ID'])
                logger.info(f"🌟 MVP for game {id_juego}: Player {mvp['PLAYER_ID']} with {mvp['PTS']} points")

                fila_equipo = df_equipos[df_equipos['TEAM_ID'] == id_equipo]
                if not fila_equipo.empty:
                    estadisticas_equipo.append(fila_equipo)
                    logger.info(f"✅ Team stats collected for game {id_juego}")
                
                time.sleep(1)
                
            except Exception as game_error:
                logger.error(f"❌ Error processing game {id_juego}: {game_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

        if not estadisticas_equipo:
            logger.error("❌ No team statistics could be retrieved")
            raise ValueError("No se pudieron recuperar estadísticas del equipo")

        # Step 3: Calculate team statistics
        logger.info("📈 Calculating team statistics...")
        df_todos = pd.concat(estadisticas_equipo, ignore_index=True)
        
        # Handle missing columns
        required_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS', 'TO']
        missing_cols = [col for col in required_cols if col not in df_todos.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns: {missing_cols}")
        
        perdidas_temp = df_todos['TO'].mean() if 'TO' in df_todos.columns else 1
        
        estadisticas = {
            'teamScore_prom50': df_todos['PTS'].mean() if 'PTS' in df_todos.columns else 0,
            'assists_prom50': df_todos['AST'].mean() if 'AST' in df_todos.columns else 0,
            'reboundsTotal_prom50': df_todos['REB'].mean() if 'REB' in df_todos.columns else 0,
            'steals_prom50': df_todos['STL'].mean() if 'STL' in df_todos.columns else 0,
            'blocks_prom50': df_todos['BLK'].mean() if 'BLK' in df_todos.columns else 0,
            'fieldGoalsPercentage_prom50': df_todos['FG_PCT'].mean() if 'FG_PCT' in df_todos.columns else 0,
            'freeThrowsPercentage_prom50': df_todos['FT_PCT'].mean() if 'FT_PCT' in df_todos.columns else 0,
            'threePointersPercentage_prom50': df_todos['FG3_PCT'].mean() if 'FG3_PCT' in df_todos.columns else 0,
            'plusMinusPoints_prom50': df_todos['PLUS_MINUS'].mean() if 'PLUS_MINUS' in df_todos.columns else 0,
            'eficiencia_ofensiva_prom50': (df_todos['PTS'].mean() if 'PTS' in df_todos.columns else 0) / (perdidas_temp + 1),
            'impacto_defensa_prom50': (df_todos['STL'].mean() if 'STL' in df_todos.columns else 0) + (df_todos['BLK'].mean() if 'BLK' in df_todos.columns else 0),
            'ratio_asistencias_turnovers_prom50': (df_todos['AST'].mean() if 'AST' in df_todos.columns else 0) / (perdidas_temp + 1),
            'eficiencia_tiro_prom50': ((df_todos['FG_PCT'].mean() if 'FG_PCT' in df_todos.columns else 0) + (df_todos['FG3_PCT'].mean() if 'FG3_PCT' in df_todos.columns else 0)) / 2,
            'home': 1
        }
        
        logger.info("✅ Team statistics calculated successfully")

        # Step 4: Get MVP statistics
        logger.info("🌟 Processing MVP statistics...")
        ids_mvp = list(set(ids_mvp))
        puntos_mvp, asistencias_mvp, rebotes_mvp, porcentaje_tiros_mvp = [], [], [], []

        for idx, id_jugador in enumerate(ids_mvp[:3]):  # Limit to 3 to reduce API calls
            logger.info(f"👤 Processing MVP {idx+1}/{min(len(ids_mvp), 3)}: Player {id_jugador}")
            
            try:
                registros = playergamelog.PlayerGameLog(player_id=id_jugador, season=temporada)
                df_mvp = registros.get_data_frames()[0].head(10)
                
                if not df_mvp.empty:
                    puntos_mvp.append(df_mvp['PTS'].mean())
                    asistencias_mvp.append(df_mvp['AST'].mean())
                    rebotes_mvp.append(df_mvp['REB'].mean())
                    porcentaje_tiros_mvp.append(df_mvp['FG_PCT'].mean())
                    logger.info(f"✅ MVP stats collected for player {id_jugador}")
                
                time.sleep(1)
                
            except Exception as mvp_error:
                logger.error(f"❌ Error fetching MVP data for player {id_jugador}: {mvp_error}")
                continue

        estadisticas['MVP_PTS'] = np.mean(puntos_mvp) if puntos_mvp else 0
        estadisticas['MVP_AST'] = np.mean(asistencias_mvp) if asistencias_mvp else 0
        estadisticas['MVP_REB'] = np.mean(rebotes_mvp) if rebotes_mvp else 0
        estadisticas['MVP_FG_PCT'] = np.mean(porcentaje_tiros_mvp) if porcentaje_tiros_mvp else 0
        estadisticas['IMPACTO_MVP'] = (estadisticas['MVP_PTS'] * 1.0 + estadisticas['MVP_AST'] * 0.7 + estadisticas['MVP_REB'] * 0.5 + 10 * 0.3)

        logger.info(f"✅ Successfully processed team {id_equipo}")
        logger.info(f"📊 Final stats preview: PTS={estadisticas['teamScore_prom50']:.1f}, AST={estadisticas['assists_prom50']:.1f}")
        return estadisticas
        
    except Exception as e:
        logger.error(f"❌ Critical error processing team {id_equipo}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@app.get("/equipos")
def obtener_equipos():
    logger.info("📋 Fetching teams list...")
    try:
        todos_equipos = nba_teams.get_teams()
        equipos = [{"id": t["id"], "name": t["full_name"]} for t in sorted(todos_equipos, key=lambda x: x["full_name"])]
        logger.info(f"✅ Retrieved {len(equipos)} teams")
        return equipos
    except Exception as e:
        logger.error(f"❌ Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predecir")
def predecir_desde_ids(entrada: EquiposInput):
    logger.info(f"🔮 Prediction request: Team {entrada.equipo1_id} vs Team {entrada.equipo2_id}")
    
    try:
        # Validate input
        if entrada.equipo1_id == entrada.equipo2_id:
            raise HTTPException(status_code=400, detail="Teams cannot be the same")
        
        logger.info("🏠 Processing home team...")
        eq1 = obtener_caracteristicas_equipo_con_mvp(entrada.equipo1_id)
        
        logger.info("✈️ Processing away team...")
        eq2 = obtener_caracteristicas_equipo_con_mvp(entrada.equipo2_id)
        eq2['home'] = 0

        logger.info("🧮 Preparing model input...")
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

        logger.info("🤖 Running prediction...")
        diferencia = np.array(v1) - np.array(v2)
        X = pd.DataFrame([diferencia], columns=[f'entreno_{k}' for k in claves_modelo])
        X_imputado = imputador.transform(X)
        X_escalado = escalador.transform(X_escalado)
        probabilidad = modelo.predict_proba(X_escalado)[0][1]

        result = {"probabilidad_victoria_local": round(probabilidad * 100, 2)}
        logger.info(f"🎯 Prediction completed: {result['probabilidad_victoria_local']}%")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    
@app.post("/predecir-estadisticas")
def predecir_con_estadisticas(datos: EstadisticasEquipos):
    logger.info("📊 Stats-based prediction request")
    try:
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

        result = {"probabilidad_victoria_local": round(probabilidad * 100, 2)}
        logger.info(f"✅ Stats prediction completed: {result['probabilidad_victoria_local']}%")
        return result
        
    except Exception as e:
        logger.error(f"❌ Stats prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}
