import { useState } from "preact/hooks";

const statLabels = {
  teamScore: "Puntos promedio",
  assists: "Asistencias promedio",
  blocks: "Bloqueos promedio",
  steals: "Robos promedio",
  turnovers: "Pérdidas promedio",
  fieldGoalsPercentage: "Porcentaje de tiros de campo",
  threePointersPercentage: "Porcentaje de triples",
  freeThrowsPercentage: "Porcentaje de tiros libres",
  reboundsTotal: "Rebotes totales",
  plusMinusPoints: "Plus/Minus puntos",
  MVP_points: "MVP puntos",
  MVP_assists: "MVP asistencias",
  MVP_rebounds: "MVP rebotes",
  MVP_fg: "MVP % tiros de campo",
};


const statKeys = Object.keys(statLabels);
const initialStats = Object.fromEntries(statKeys.map((stat) => [stat, ""]));

export default function PredictPage() {
  const [teamLocal, setTeamLocal] = useState({ ...initialStats });
  const [teamVisitor, setTeamVisitor] = useState({ ...initialStats });
  const [winProbability, setWinProbability] = useState(null);

  const handleChange = (setter) => (field, value) => {
    setter((prev) => ({ ...prev, [field]: value }));
  };

  const buildTeamData = (team, isHome) => {
    const data = Object.fromEntries(
      statKeys.map((field) => [field, parseFloat(team[field])])
    );

    data.eficienciaOfensiva = data.teamScore / (data.turnovers + 1);
    data.impactoDefensa = data.steals + data.blocks;
    data.ratioAsistenciasTurnovers = data.assists / (data.turnovers + 1);
    data.eficienciaTiro = (data.fieldGoalsPercentage + 1.5 * data.threePointersPercentage) / 2;
    data.impactoMVP = 
      data.MVP_points * 1 + 
      data.MVP_assists * 0.7 + 
      data.MVP_rebounds * 0.5 + 
      10 * 0.3;

    data.isHome = isHome;

    return data;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();


    const eq1Data = buildTeamData(teamLocal, 1);
    const eq2Data = buildTeamData(teamVisitor, 0);

    const res = await fetch("/api/predecir-estadisticas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ equipo1: eq1Data, equipo2: eq2Data }),
    });

    const data = await res.json();
    setWinProbability(data.probabilidad_victoria_local);
};

  const TeamForm = ({ team, setTeam, title }) => (
    <div class="form-box">
      <h2 class="form-title">{title}</h2>
      {statKeys.map((field) => (
        <div class="form-group">
          <label class="form-label">{statLabels[field]}</label>
          <input
            type="text"
            required
            placeholder={statLabels[field]}
            class="form-input"
            value={team[field]}
            onChange={(e) => handleChange(setTeam)(field, e.target.value)}
          />
        </div>
      ))}
    </div>
  );

  return (
    <div class="container">
      <h1 class="title">¿Quién ganará?</h1>
      <form onSubmit={handleSubmit}>
        <div class="form">
          <TeamForm team={teamLocal} setTeam={setTeamLocal} title="🏠 Equipo Local" />
          <TeamForm team={teamVisitor} setTeam={setTeamVisitor} title="🛫 Equipo Visitante" />
        </div>
        <div class="button-container">
          <button type="submit" class="submit-button">
            Predecir
          </button>
        </div>
      </form>

      {winProbability !== null && !isNaN(winProbability) && (
        <div class="result-box">
          <h3 class="result-title">Resultado de la Predicción</h3>
          <p class="result-text">
            Probabilidad de victoria del equipo <span class="highlight">local</span>:{" "}
            <span class="percentage">{winProbability.toFixed(2)}%</span>
          </p>
        </div>
      )}
    </div>
  );
}
