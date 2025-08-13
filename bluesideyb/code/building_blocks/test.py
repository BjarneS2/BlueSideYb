import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity, kron
from IonQubit import System, IonQubit  # replace with actual module name


def test_ket_bra_proj_xop():
    sys = System(dimension=4)
    for i in range(4):
        ket_i = sys.ket(i)
        bra_i = sys.bra(i)
        proj_i = sys.proj(i)
        xop_i = sys.xop(i, (i+1)%4)

        # ket[i] @ bra[i] = proj[i]
        assert np.allclose((ket_i @ bra_i).todense(), proj_i.todense())
        # xop correctness
        assert np.allclose((ket_i @ sys.bra((i+1)%4)).todense(), xop_i.todense())

def test_vec_unvec():
    sys = System(dimension=3)
    rho = csr_matrix(np.eye(3))
    v = sys.vec(rho)
    rho_rec = sys.unvec(v, 3)
    assert np.allclose(rho.todense(), rho_rec.todense())

def test_liouvillian():
    sys = System(dimension=2)
    H = csr_matrix([[1,0],[0,-1]])
    C = [csr_matrix([[0,1],[0,0]])]
    L = sys.liouvillian(H, C)
    assert L.shape == (4,4)
    # Liouvillian should be complex
    assert np.iscomplexobj(L.todense())


@pytest.mark.parametrize("state", [None, 0, 1, '0','1','+','-','i','-i'])
def test_initialization(state):
    # testing if proper quantum states are being created
    iq = IonQubit(initial_state=state)
    assert iq.state.shape == (4,4)
    assert np.isclose(iq.state.diagonal().sum(), 1.0)  # density matrix trace=1

def test_Hamiltonians():
    iq = IonQubit()
    H0 = iq.H_intrinsic()
    Hraman = iq.H_raman(Omega1=1.0, phi1=0.0, Omega2=0.5, phi2=np.pi/2)
    Hdet = iq.H_detect(Omega_det=0.8)
    # All should be 4x4 sparse matrices
    # Is not testing if it is actually what you want from it - that might come in later on though
    for H in [H0, Hraman, Hdet]:
        assert H.shape == (4,4)

def test_collapse_operators():
    iq = IonQubit()
    C_ops = [
        iq.C_P_to_0(1.0), iq.C_P_to_1(0.5), iq.C_P_to_D(0.2),
        iq.C_D_to_0(0.3), iq.C_D_to_1(0.4),
        iq.C_dephase(0.1), iq.C_flip_up(0.05), iq.C_flip_dn(0.05)
    ]
    for C in C_ops:
        assert C.shape == (4,4)

def test_step_evolution():
    iq = IonQubit(initial_state='0')
    H = iq.H_intrinsic()
    C_ops = []
    L = iq.build_L(H, C_ops)
    rho_next = iq.step(iq.state, L, dt=0.01)
    assert rho_next.shape == (4,4)
    # trace should remain 1 for unitary evolution
    assert np.isclose(rho_next.diagonal().sum(), 1.0)

def test_expectations_and_bloch():
    iq = IonQubit(initial_state='+')
    exps = iq.qubit_expectations()
    for key in ['0','1','+','-','i','-i']:
        assert 0.0 <= exps[key] <= 1.0

    bloch = iq.bloch_vector()
    assert bloch.shape == (3,)
    assert np.all(np.abs(bloch) <= 1.0)  # Bloch vector norm <= 1


if __name__ == "__main__":
    import sys as s
    s.exit(pytest.main([__file__]))