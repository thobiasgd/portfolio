import openvsp as vsp
import numpy as np
import math
import os
from pandas import read_csv
from utils import get_base_dir

class Aircraft:
    def __init__(self, genes, generation=0, index=0):
        """Representa uma aeronave definida por cromossomos (genes)."""
        self.genes = genes  # [envergadura, corda_root, corda_tip, sweep1, sweep2, incidencia]
        self.generation = generation
        self.index = index
        self.fitness = 0.0
        self.results = {}

    # --------------------------------------------------
    def modelar(self):
        """Cria o modelo geométrico no OpenVSP."""
        vsp.VSPRenew()
        wing_id = vsp.AddGeom("WING")
        vsp.SetGeomName(wing_id, f"Wing_G{self.generation}_I{self.index}")

        enverg, c_root, c_tip, sweep1, sweep2, incidencia = self.genes

        # Define as seções da asa
        vsp.SetDriverGroup(wing_id, 1, vsp.AR_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
        vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", c_root)
        vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_1", c_root)
        vsp.SetParmVal(wing_id, "Root_Chord", "XSec_2", c_root)
        vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_2", c_tip)
        vsp.SetParmVal(wing_id, "Span", "XSec_1", enverg/2)
        vsp.SetParmVal(wing_id, "Span", "XSec_2", enverg/2)
        vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep1)
        vsp.SetParmVal(wing_id, "Sweep", "XSec_2", sweep2)
        vsp.SetParmVal(wing_id, "Twist", "XSec_0", incidencia)
        vsp.Update()

        # Salva o arquivo VSP
        dir_path = get_base_dir() / f"G{self.generation}_I{self.index}"
        os.makedirs(dir_path, exist_ok=True)
        self.model_path = dir_path / f"aircraft_G{self.generation}_I{self.index}.vsp3"
        vsp.WriteVSPFile(str(self.model_path))

    # --------------------------------------------------
    def simular(self):
        """Executa simulação VSPAERO e coleta resultados."""
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(str(self.model_path))
        vsp.Update()

        analysis = "VSPAEROSweep"
        vsp.SetIntAnalysisInput(analysis, "GeomSet", [0])
        vsp.SetDoubleAnalysisInput(analysis, "AlphaStart", [-3])
        vsp.SetDoubleAnalysisInput(analysis, "AlphaEnd", [15])
        vsp.SetIntAnalysisInput(analysis, "AlphaNpts", [19])
        vsp.SetDoubleAnalysisInput(analysis, "ReCref", [5e5])
        vsp.SetDoubleAnalysisInput(analysis, "Rho", [1.225])
        vsp.SetDoubleAnalysisInput(analysis, "Vinf", [14.2])
        vsp.Update()

        res_id = vsp.ExecAnalysis(analysis)
        vsp.PrintResults(res_id)

        # lê arquivo polar gerado
        polar_path = str(self.model_path).replace(".vsp3", "_DegenGeom.polar")
        if not os.path.exists(polar_path):
            self.fitness = 0
            return

        df = read_csv(polar_path, sep=r"\s+")
        CL = df["CL"].to_numpy()
        CD = df["CD"].to_numpy()
        CLmax = np.max(CL)
        CDmin = np.min(CD)
        self.results = {"CLmax": CLmax, "CDmin": CDmin}
