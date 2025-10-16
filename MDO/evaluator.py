import math

def evaluate(aircraft):
    """Função de avaliação multiobjetivo simples."""
    if not aircraft.results:
        aircraft.fitness = 0
        return 0

    CLmax = aircraft.results["CLmax"]
    CDmin = aircraft.results["CDmin"]

    # parâmetros fixos
    rho = 1.225
    Sref = aircraft.genes[0] * aircraft.genes[1]
    MTOW = 15 * 9.81
    g = 9.81
    T = 43
    mu = 0.035

    Vstall = math.sqrt((2 * MTOW) / (rho * Sref * CLmax))
    Vto = 1.2 * Vstall
    D = 0.5 * rho * Vto**2 * Sref * CDmin
    L = 0.5 * rho * Vto**2 * Sref * CLmax
    F = T - (D + mu * (MTOW - L))

    dist = (1.44 * MTOW**2) / (g * rho * Sref * CLmax * F)
    fitness = (CLmax / CDmin) / dist
    aircraft.fitness = max(fitness, 0)
    return aircraft.fitness
