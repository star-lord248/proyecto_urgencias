// =====================================================================
// Simulación de eventos discretos (Monte Carlo) de una sala de urgencias
// Cola M/M/c con prioridades de triaje.
//
// VERSIÓN CON ENTRADA DE DATOS: el programa pregunta los valores por
// teclado. Si presionas Enter sin escribir nada, usa el valor por defecto
// que aparece entre corchetes [ ].
//
// Compilar:  g++ -O2 -std=c++17 -o urgencias simulacion_urgencias.cpp
// Ejecutar:  ./urgencias          (genera resultados_cpp.csv y un resumen)
//
// Unidades de tiempo: minutos.
// Nivel 1 = más crítico ... Nivel L = menos urgente.
// =====================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------
// 1. PARÁMETROS (ahora son variables que se leen en main)
// ---------------------------------------------------------------------
double LAMBDA_H = 10.0;                 // llegadas por hora
double LAMBDA = LAMBDA_H / 60.0;        // llegadas por minuto
double T_TOTAL = 30.0 * 24.0 * 60.0;    // horizonte de simulación (min)
double T_CALENTAMIENTO = 24.0 * 60.0;   // se descarta este tiempo inicial (min)
int REPLICAS = 30;                      // réplicas por escenario
uint64_t SEMILLA_BASE = 20261109ULL;

struct Escenario {
    string nombre;
    int c;                    // número de médicos
    bool preemptivo;          // true: un crítico interrumpe a uno menos grave
    vector<double> prob;      // probabilidad de cada nivel de triaje
    vector<double> media;     // tiempo medio de atención por nivel (min)
    bool validaErlang;        // true si existe fórmula teórica (Erlang C)
};

// ---------------------------------------------------------------------
// 2. LECTURA DE DATOS POR TECLADO
// ---------------------------------------------------------------------
template <class T>
T leer(const string& msg, T def) {
    cout << msg << " [" << def << "]: ";
    string linea;
    getline(cin, linea);
    if (linea.empty()) return def;
    stringstream ss(linea);
    T x;
    if (ss >> x) return x;
    cout << "  Valor inválido, se usa " << def << "\n";
    return def;
}

// Lee n números separados por espacios (o Enter para el valor por defecto)
vector<double> leerVector(const string& msg, const vector<double>& def) {
    cout << msg << " [";
    for (size_t i = 0; i < def.size(); ++i) cout << (i ? " " : "") << def[i];
    cout << "]: ";
    string linea;
    getline(cin, linea);
    if (linea.empty()) return def;
    stringstream ss(linea);
    vector<double> v;
    double x;
    while (ss >> x) v.push_back(x);
    if (v.size() != def.size()) {
        cout << "  Se esperaban " << def.size() << " valores, se usan los de por defecto\n";
        return def;
    }
    return v;
}

vector<int> leerEnteros(const string& msg, const vector<int>& def) {
    cout << msg << " [";
    for (size_t i = 0; i < def.size(); ++i) cout << (i ? " " : "") << def[i];
    cout << "]: ";
    string linea;
    getline(cin, linea);
    if (linea.empty()) return def;
    stringstream ss(linea);
    vector<int> v;
    int x;
    while (ss >> x)
        if (x > 0) v.push_back(x);
    if (v.empty()) {
        cout << "  Valores inválidos, se usan los de por defecto\n";
        return def;
    }
    return v;
}

// ---------------------------------------------------------------------
// 3. GENERADOR DE VARIABLES ALEATORIAS (transformada inversa)
// ---------------------------------------------------------------------
struct Generador {
    mt19937_64 gen;
    uniform_real_distribution<double> U{0.0, 1.0};

    explicit Generador(uint64_t semilla) : gen(semilla) {}

    double uniforme() { return U(gen); }

    // Exponencial por transformada inversa:
    // F(x) = 1 - e^(-x/media)  =>  x = -media * ln(1 - u)
    double exponencial(double media) { return -media * log(1.0 - uniforme()); }

    // Variable discreta: nivel de triaje (0..L-1) con probabilidades acumuladas
    int nivel(const vector<double>& prob) {
        double u = uniforme(), acum = 0.0;
        for (size_t i = 0; i < prob.size(); ++i) {
            acum += prob[i];
            if (u < acum) return static_cast<int>(i);
        }
        return static_cast<int>(prob.size()) - 1;
    }
};

// ---------------------------------------------------------------------
// 4. ESTRUCTURAS DE LA SIMULACIÓN
// ---------------------------------------------------------------------
enum class Tipo { LLEGADA, SALIDA };

struct Evento {
    double t;        // instante del evento
    Tipo tipo;
    int servidor;    // solo para SALIDA
    long token;      // para invalidar salidas de pacientes expulsados
    bool operator>(const Evento& o) const { return t > o.t; }
};

struct Paciente {
    double llegada;
    int nivel;       // 0 = crítico
    bool iniciado;   // ya fue atendido al menos una vez
};

struct Servidor {
    int paciente = -1;   // -1 = libre
    long token = 0;
};

// ---------------------------------------------------------------------
// 5. UNA RÉPLICA DE LA SIMULACIÓN
//    Devuelve, por nivel, la lista de esperas (llegada -> primer inicio de
//    atención) de los pacientes que llegaron después del calentamiento.
// ---------------------------------------------------------------------
vector<vector<double>> simular(const Escenario& e, uint64_t semilla) {
    const int L = static_cast<int>(e.prob.size());
    Generador g(semilla);

    vector<Paciente> pac;
    vector<deque<int>> colas(L);          // una cola FIFO por nivel
    vector<Servidor> serv(e.c);
    vector<vector<double>> esperas(L);
    priority_queue<Evento, vector<Evento>, greater<Evento>> fel;  // lista de eventos futuros

    // Primera llegada
    fel.push({g.exponencial(1.0 / LAMBDA), Tipo::LLEGADA, -1, 0});

    // El servidor s empieza (o reanuda) la atención del paciente p en el instante t
    auto iniciarAtencion = [&](int s, int p, double t) {
        Paciente& P = pac[p];
        if (!P.iniciado) {
            P.iniciado = true;
            if (P.llegada >= T_CALENTAMIENTO) esperas[P.nivel].push_back(t - P.llegada);
        }
        serv[s].paciente = p;
        serv[s].token++;   // invalida cualquier salida anterior de este servidor
        // Por la propiedad sin memoria de la exponencial, un paciente que
        // reanuda tras ser expulsado necesita un tiempo exp(media) nuevo.
        fel.push({t + g.exponencial(e.media[P.nivel]), Tipo::SALIDA, s, serv[s].token});
    };

    while (!fel.empty()) {
        Evento ev = fel.top();
        fel.pop();
        const double t = ev.t;

        if (ev.tipo == Tipo::LLEGADA) {
            // Programar la siguiente llegada (proceso de Poisson)
            double prox = t + g.exponencial(1.0 / LAMBDA);
            if (prox <= T_TOTAL) fel.push({prox, Tipo::LLEGADA, -1, 0});

            int nivel = g.nivel(e.prob);
            pac.push_back({t, nivel, false});
            int p = static_cast<int>(pac.size()) - 1;

            // ¿Hay un médico libre?
            int libre = -1;
            for (int s = 0; s < e.c; ++s)
                if (serv[s].paciente == -1) { libre = s; break; }

            if (libre != -1) {
                iniciarAtencion(libre, p, t);
            } else if (e.preemptivo) {
                // Buscar al médico que atiende al paciente de peor prioridad,
                // siempre que sea estrictamente peor que el que llega
                int peor = -1, nivelPeor = nivel;
                for (int s = 0; s < e.c; ++s) {
                    int nv = pac[serv[s].paciente].nivel;
                    if (nv > nivelPeor) { nivelPeor = nv; peor = s; }
                }
                if (peor != -1) {
                    int expulsado = serv[peor].paciente;
                    colas[nivelPeor].push_front(expulsado);  // vuelve al frente de su cola
                    iniciarAtencion(peor, p, t);
                } else {
                    colas[nivel].push_back(p);
                }
            } else {
                colas[nivel].push_back(p);
            }
        } else {  // SALIDA
            Servidor& S = serv[ev.servidor];
            if (ev.token != S.token || S.paciente == -1) continue;  // evento invalidado
            S.paciente = -1;
            // Tomar al paciente de mayor prioridad en cola (FIFO dentro del nivel)
            for (int n = 0; n < L; ++n) {
                if (!colas[n].empty()) {
                    int p = colas[n].front();
                    colas[n].pop_front();
                    iniciarAtencion(ev.servidor, p, t);
                    break;
                }
            }
        }
    }
    return esperas;
}

// ---------------------------------------------------------------------
// 6. TEORÍA: tiempo de espera medio en cola de un M/M/c (Erlang C)
// ---------------------------------------------------------------------
double erlangC_Wq(double lambda, double mu, int c) {
    double a = lambda / mu;        // carga ofrecida (erlangs)
    double rho = a / c;            // utilización
    double termino = 1.0, suma = 0.0;
    for (int k = 0; k < c; ++k) {  // suma de a^k/k! para k = 0..c-1
        suma += termino;
        termino *= a / (k + 1);
    }
    // aquí termino = a^c / c!
    double cola = termino / (1.0 - rho);
    double Pw = cola / (suma + cola);   // probabilidad de tener que esperar
    return Pw / (c * mu - lambda);      // Wq
}

// Valor t de Student (95 %, dos colas) según los grados de libertad
double tStudent(int gl) {
    static const double tabla[30] = {
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
    if (gl < 1) return 0.0;
    if (gl <= 30) return tabla[gl - 1];
    return 1.96;  // aproximación normal para muchas réplicas
}

// Media e intervalo de confianza (95 %) de un vector de resultados por réplica
pair<double, double> ic(const vector<double>& v) {
    double n = static_cast<double>(v.size());
    double m = accumulate(v.begin(), v.end(), 0.0) / n;
    if (v.size() < 2) return {m, 0.0};
    double ss = 0.0;
    for (double x : v) ss += (x - m) * (x - m);
    double s = sqrt(ss / (n - 1.0));
    return {m, tStudent(static_cast<int>(v.size()) - 1) * s / sqrt(n)};
}

// ---------------------------------------------------------------------
// 7. PROGRAMA PRINCIPAL: pide datos, corre escenarios, CSV y resumen
// ---------------------------------------------------------------------
int main() {
    cout << "=== Simulacion de sala de urgencias ===\n";
    cout << "(Enter = usar el valor por defecto entre corchetes)\n\n";

    // ---- Parámetros generales ----
    LAMBDA_H = leer<double>("Llegadas por hora (lambda)", 10.0);
    LAMBDA = LAMBDA_H / 60.0;
    double dias = leer<double>("Dias a simular por replica", 30.0);
    double diasCal = leer<double>("Dias de calentamiento (se descartan)", 1.0);
    T_TOTAL = dias * 24.0 * 60.0;
    T_CALENTAMIENTO = diasCal * 24.0 * 60.0;
    REPLICAS = leer<int>("Numero de replicas por escenario", 30);
    if (REPLICAS < 2) REPLICAS = 2;
    SEMILLA_BASE = leer<uint64_t>("Semilla base", 20261109ULL);

    // ---- Niveles de triaje ----
    int L = leer<int>("Numero de niveles de triaje", 5);
    if (L < 1) L = 1;

    vector<double> defProb = {0.05, 0.15, 0.35, 0.30, 0.15};
    vector<double> defMedia = {30, 25, 20, 15, 10};
    defProb.resize(L, 1.0 / L);
    defMedia.resize(L, 15.0);
    if (L != 5) {  // si cambia el número de niveles, repartir por defecto de forma pareja
        defProb.assign(L, 1.0 / L);
    }

    vector<double> PROB = leerVector("Probabilidad de cada nivel (nivel 1 = critico)", defProb);
    vector<double> MEDIA = leerVector("Tiempo medio de atencion por nivel (min)", defMedia);

    double sp = accumulate(PROB.begin(), PROB.end(), 0.0);
    if (fabs(sp - 1.0) > 1e-6 && sp > 0) {
        cout << "  Las probabilidades sumaban " << sp << ", se normalizan para que sumen 1\n";
        for (double& p : PROB) p /= sp;
    }

    // ---- Médicos y tipos de escenario ----
    vector<int> medicos = leerEnteros("Numeros de medicos a comparar (separados por espacio)", {4, 5, 6});
    int conValid = leer<int>("Incluir escenarios de validacion contra Erlang C? (1 = si, 0 = no)", 1);

    double ES = 0.0;  // tiempo medio de atención global
    for (size_t i = 0; i < PROB.size(); ++i) ES += PROB[i] * MEDIA[i];

    vector<Escenario> escenarios;
    for (int c : medicos) {
        string k = "c" + to_string(c);
        if (conValid) {
            // Validación 1: un solo tipo de paciente = M/M/c puro
            escenarios.push_back({k + "_base_MMc", c, false, {1.0}, {ES}, true});
            // Validación 2: prioridades con igual tiempo de servicio -> la espera
            // global debe coincidir con la de Erlang C (ley de conservación)
            escenarios.push_back({k + "_igual_NP", c, false, PROB, vector<double>(L, ES), true});
        }
        // Escenarios reales: prioridad no preemptiva y preemptiva
        escenarios.push_back({k + "_NP", c, false, PROB, MEDIA, false});
        escenarios.push_back({k + "_P", c, true, PROB, MEDIA, false});
    }

    // ---- Aviso de estabilidad ----
    cout << "\nTiempo medio de atencion global E[S] = " << ES << " min\n";
    for (int c : medicos) {
        double rho = (LAMBDA * ES) / c;
        cout << "  c = " << c << " medicos -> utilizacion rho = " << rho;
        if (rho >= 1.0) cout << "  (!) INESTABLE: la cola crece sin limite";
        cout << "\n";
    }
    cout << "\n";

    ofstream csv("resultados_cpp.csv");
    csv << fixed << setprecision(4);
    csv << "escenario,replica,nivel,n,espera_prom,espera_p95\n";
    cout << fixed << setprecision(3);
    cout << "lambda = " << LAMBDA_H << " pac/h | E[S] global = " << ES << " min\n";

    for (const auto& e : escenarios) {
        const int Le = static_cast<int>(e.prob.size());
        vector<vector<double>> prom(Le), p95(Le);
        vector<double> global;

        for (int r = 0; r < REPLICAS; ++r) {
            auto esperas = simular(e, SEMILLA_BASE + r);
            double suma = 0.0;
            size_t total = 0;
            for (int n = 0; n < Le; ++n) {
                auto& v = esperas[n];
                if (v.empty()) continue;
                double s = accumulate(v.begin(), v.end(), 0.0);
                sort(v.begin(), v.end());
                size_t idx = static_cast<size_t>(ceil(0.95 * v.size())) - 1;  // percentil 95
                double m = s / v.size();
                prom[n].push_back(m);
                p95[n].push_back(v[idx]);
                csv << e.nombre << ',' << r + 1 << ',' << n + 1 << ',' << v.size()
                    << ',' << m << ',' << v[idx] << '\n';
                suma += s;
                total += v.size();
            }
            if (total > 0) global.push_back(suma / total);
        }

        cout << "\n== " << e.nombre << " (c=" << e.c
             << (e.preemptivo ? ", preemptivo" : ", no preemptivo") << ") ==\n";
        if (Le > 1) {
            for (int n = 0; n < Le; ++n) {
                if (prom[n].empty()) {
                    cout << "  Nivel " << n + 1 << ": sin pacientes\n";
                    continue;
                }
                auto [m, h] = ic(prom[n]);
                auto [mp, hp] = ic(p95[n]);
                cout << "  Nivel " << n + 1 << ": espera prom = " << m << " +- " << h
                     << " min | p95 = " << mp << " min\n";
            }
        }
        if (!global.empty()) {
            auto [mg, hg] = ic(global);
            cout << "  Global : espera prom = " << mg << " +- " << hg << " min\n";
        }
        if (e.validaErlang && (LAMBDA * ES) / e.c < 1.0)
            cout << "  Wq teorico (Erlang C) = " << erlangC_Wq(LAMBDA, 1.0 / ES, e.c) << " min\n";
    }
    cout << "\nResultados por replica guardados en resultados_cpp.csv\n";
    return 0;
}