
import json
from predict_api import obtener_caracteristicas_equipo_con_mvp
from nba_api.stats.static import teams

datos = {}
equipos = teams.get_teams()

for equipo in equipos:
    id_equipo = equipo["id"]
    nombre = equipo["full_name"]
    try:
        print(f"Procesando: {nombre}")
        datos[id_equipo] = obtener_caracteristicas_equipo_con_mvp(id_equipo)
    except Exception as e:
        print(f"❌ Error con {nombre}: {e}")

with open("estadisticas_precalculadas.json", "w") as f:
    json.dump(datos, f)

print("✅ Archivo generado: estadisticas_precalculadas.json")
