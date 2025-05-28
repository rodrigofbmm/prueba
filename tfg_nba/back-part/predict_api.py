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
        juegos = teamgamelog.TeamGameLog(team_id=id_equipo, season="2024-25", timeout=8).get_data_frames()[0].head(10)
        estadisticas = []
        
        for j in juegos['Game_ID']:
            try:
                box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=j, timeout=5)
                equipos = box.get_data_frames()[1]
                fila = equipos[equipos['TEAM_ID'] == id_equipo]
                if not fila.empty:
                    estadisticas.append(fila)
                time.sleep(0.5)
            except:
                continue
                
        if not estadisticas:
            raise ValueError("Sin datos")
            
        df = pd.concat(estadisticas)
        return {
            'teamScore_prom50': df['PTS'].mean(),
            'assists_prom50': df['AST'].mean(), 
            'reboundsTotal_prom50': df['REB'].mean(),
            'steals_prom50': df['STL'].mean(),
            'blocks_prom50': df['BLK'].mean(),
            'fieldGoalsPercentage_prom50': df['FG_PCT'].mean(),
            'freeThrowsPercentage_prom50': df['FT_PCT'].mean(),
            'threePointersPercentage_prom50': df['FG3_PCT'].mean(),
            'plusMinusPoints_prom50': df['PLUS_MINUS'].mean(),
            'home': 1
        }
    except:
        raise

def obtener_estadisticas_equipo(id_equipo):
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(obtener_datos_api, id_equipo).result(timeout=15)
    except:
        return cache.get(str(id_equipo), {})

@app.get("/equipos")
def equipos():
    try:
        return sorted([{"id": t["id"], "name": t["full_name"]} for t in nba_teams.get_teams()], key=lambda x: x["name"])
    except:
        return []

@app.post("/predecir")
def predecir_ids(input: EquiposInput):
    try:
        eq1 = obtener_estadisticas_equipo(input.equipo1_id)
        eq2 = obtener_estadisticas_equipo(input.equipo2_id)
        
        if not eq1 or not eq2:
            return {"error": "No se pudieron obtener estadísticas"}
            
        eq2["home"] = 0
        claves = list(eq1.keys())
        
        dif = np.array([eq1.get(k, 0) - eq2.get(k, 0) for k in claves])
        X = pd.DataFrame([dif], columns=[f'entreno_{k}' for k in claves])
        X_final = escalador.transform(imputador.transform(X))
        
        prob = modelo.predict_proba(X_final)[0][1] * 100
        return {"probabilidad_victoria_local": round(prob, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predecir-estadisticas")  
def predecir_estadisticas(data: EstadisticasEquipos):
    try:
        mapeo = {
            'teamScore': 'teamScore_prom50',
            'assists': 'assists_prom50', 
            'blocks': 'blocks_prom50',
            'steals': 'steals_prom50',
            'fieldGoalsPercentage': 'fieldGoalsPercentage_prom50',
            'threePointersPercentage': 'threePointersPercentage_prom50',
            'freeThrowsPercentage': 'freeThrowsPercentage_prom50',
            'reboundsTotal': 'reboundsTotal_prom50',
            'plusMinusPoints': 'plusMinusPoints_prom50',
            'isHome': 'home'
        }
        
        eq1 = {mapeo[k]: v for k, v in data.equipo1.items() if k in mapeo}
        eq2 = {mapeo[k]: v for k, v in data.equipo2.items() if k in mapeo}
        
        claves = list(mapeo.values())
        dif = np.array([eq1.get(k, 0) - eq2.get(k, 0) for k in claves])
        
        X = pd.DataFrame([dif], columns=[f'entreno_{k}' for k in claves])
        X_final = escalador.transform(imputador.transform(X))
        
        prob = modelo.predict_proba(X_final)[0][1] * 100
        return {"probabilidad_victoria_local": round(prob, 2)}
    except Exception as e:
        return {"error": str(e)}
