import { useEffect, useState } from "preact/hooks";

type Team = {
  id: number;
  name: string;
};

export default function PredictByTeamIDs() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [localId, setLocalId] = useState<number | null>(null);
  const [visitorId, setVisitorId] = useState<number | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false); // Nuevo estado

  useEffect(() => {
    fetch("http://localhost:8008/equipos")
      .then((res) => res.json())
      .then((data: Team[]) => {
        setTeams(data);
      })
  }, []);

  const handleSubmit = async (e: Event) => {
    e.preventDefault();

    if (localId === null || visitorId === null) {
      alert("Selecciona ambos equipos.");
      return;
    }

    if (localId === visitorId) {
      alert("Los equipos no pueden ser iguales.");
      return;
    }

    setIsLoading(true);
    setProbability(null);

    const res = await fetch("http://localhost:8008/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        equipo1_id: localId,
        equipo2_id: visitorId,
      }),
    });

    const data = await res.json();
    console.log("Respuesta del servidor:", data);
    setProbability(data.probabilidad_victoria_local);
    setIsLoading(false);

  };

  return (
    <div class="prediction-container">
      <h1 class="prediction-title">🏀 Predicción de Partido NBA</h1>
      <form onSubmit={handleSubmit} class="prediction-form">
        <div class="prediction-form-group">
          <label class="prediction-label">🏠 Equipo Local</label>
          <select
            class="prediction-select"
            value={localId ?? ""}
            onChange={(e) => setLocalId(Number(e.currentTarget.value))}
            disabled={isLoading}
          >
            <option value="" disabled>Selecciona equipo local</option>
            {teams.map((team) => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </div>

        <div class="prediction-form-group">
          <label class="prediction-label">✈️ Equipo Visitante</label>
          <select
            class="prediction-select"
            value={visitorId ?? ""}
            onChange={(e) => setVisitorId(Number(e.currentTarget.value))}
            disabled={isLoading}
          >
            <option value="" disabled>Selecciona equipo visitante</option>
            {teams.map((team) => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </div>

        <button type="submit" class="prediction-submit" disabled={isLoading} >
          {isLoading ? "⏳ Procesando..." : "🔮 Predecir resultado"}
        </button>
      </form>

      {probability !== null && !isLoading && (
        <div class="prediction-result">
          <p class="prediction-result-text">
            🏆 Probabilidad de victoria del equipo local:
          </p>
          <span class="prediction-percentage">
            {probability?.toFixed(2) ?? 'N/A'}%
          </span>
        </div>
      )}
    </div>
  );
}
