import numpy as np
from pyscf import gto, scf, fci, ao2mo
import itertools
import scipy.special

mol = gto.Mole()
R = 1.2
get_atom_Hn = lambda n: [['H', [R * i, 0.0, 0.0]] for i in range(n)]
mol.build(
    atom=get_atom_Hn(8),
    basis='sto-3g',
    symmetry=True,
)

hf = scf.RHF(mol)
hf.kernel()
hf.mo_coeff = np.array(hf.mo_coeff)
cisolver = fci.FCI(mol, hf.mo_coeff)
EFCI, fci_vec = cisolver.kernel(davidson_only=False)
print('FCI energy', EFCI)


def AO2MO_1e(O, mo_coeff):
    return mo_coeff.T.dot(O).dot(mo_coeff)


nocc = mol.nelectron // 2
nall = mol.nao
nvir = nall - nocc
N = scipy.special.comb(nall, nocc, exact=True)
print('Number of determinants', N**2)

h1 = AO2MO_1e(hf.get_hcore(), hf.mo_coeff)
eri = ao2mo.kernel(mol, hf.mo_coeff)
eri_full = cisolver.absorb_h1e(h1, eri, nall, (nocc, nocc), 0.5)
contract = lambda fci_vec: cisolver.contract_2e(eri_full, fci_vec, nall, (nocc, nocc))
res = contract(fci_vec) - fci_vec * (EFCI - mol.energy_nuc())
print('FCI residual', np.linalg.norm(res))

occs = np.array(list(itertools.combinations(range(nall), nocc)), dtype=int)
occs_pyscf = (nall - occs - 1)[::-1, ::-1]

vec0 = np.zeros((N, N))
vec0[0, 0] = 1
H00 = contract(vec0)[0, 0]

E_mo = np.sum(hf.mo_energy[occs_pyscf], axis=1)
Ediag = (E_mo[:, None] + E_mo[None, :])
Ediag += H00 - Ediag[0, 0]
Ediff = Ediag[0, 0] - Ediag
Ediff[0, 0] = np.inf

apply_V = lambda vec: contract(vec) - vec * Ediag
V0 = apply_V(vec0)

psi_pert = {}
psi_pert[0] = vec0
E_pert = {}
E_pert[0] = Ediag[0, 0]

max_order = 20
# recursion relation of Rayleigh-Schrödinger perturbation theory
for k in range(1, max_order + 1):
    E_pert[k] = np.sum(psi_pert[k - 1] * V0)
    print('MP', k, 'total energy', sum(E_pert.values()) + mol.energy_nuc())
    tmp_psi = apply_V(psi_pert[k - 1])
    for j in range(1, k):
        tmp_psi -= E_pert[j] * psi_pert[k - j]
    psi_pert[k] = tmp_psi / Ediff
