export default function Home() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="text-center max-w-xl px-6">
        <h1 className="text-3xl font-bold mb-4">🏀 NBA Match Predictor</h1>
        <p className="mb-6 text-lg">
          Bienvenido al <strong>Laboratorio de Predicción NBA</strong>. Aquí puedes explorar dos modos:
        </p>
        <ul className="list-disc list-inside space-y-2 mb-8 text-left">
          <li><strong>Dos equipos:</strong>: selecciona <span className="text-blue-600">dos equipos de la NBA</span> para obtener una predicción automática basada en sus últimos partidos.</li>
          <li><strong>Laboratorio manual:</strong>: ingresa tus propias estadísticas para experimentar y ver cómo cambian las probabilidades de victoria.</li>
        </ul>
      </div>
    </div>
  );
}
