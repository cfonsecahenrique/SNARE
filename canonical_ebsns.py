"""
Canonical EBSNs under the symmetry of the EBSNs paper (Methods, Eq. 1).

The symmetry group has order 4:
    - identity
    - paper-mirror  : G <-> B swap on the recipient, with output inversion
                      (Eq. 1: d'_X1Y = 1 - d_X0Y for X in {C,D}, Y in {O,M})
    - EP-swap       : Cooperative <-> Competitive emotion-profile labels
    - mirror + EP-swap

Trivial (non-emotion-discriminating) norms - those where d_XYO == d_XYM at
every (action, recipient) - are excluded by default; they correspond to the
``Emotion_Leniency == 1'' filter used elsewhere in this codebase.

Flatten convention (matching ``aux_functions.ebsn_to_GB``):

    position  0  1  2  3  4  5  6  7
              DB DB DG DG CB CB CG CG
              M  O  M  O  M  O  M  O

    where action D=0, C=1; recipient reputation B=0, G=1;
    emotion profile M=0 (Competitive), O=1 (Cooperative).

Tie-break heuristics for choosing the canonical representative of an orbit
(permissive-on-cooperative convention, paper Methods):

    (0)   prefer the orbit member that places the more permissive 2nd-order
          rule (the half with more 1-outputs) on the cooperative-emotion side,
          i.e. maximise ``coop_ones(d) - comp_ones(d)``.
          Justification: the cooperative emotional profile is defined
          (de Melo et al. 2021) as the leniency-signalling profile, so we
          display the more permissive 2nd-order rule on that side.
    (i)   prefer bits[7]=1   (CGO=1)         -- Ohtsuki-Iwasa style tie-break
    (ii)  prefer bits[5]=1   (CBO=1)
    (iii) prefer bits[1]=1   (DBO=1)
    (iv)  lex-greatest        (deterministic final tie-break)
"""

from typing import List, Set, Tuple

Bits = Tuple[int, ...]

# ---------------------------------------------------------------------------
# Symmetry operations
# ---------------------------------------------------------------------------
def paper_mirror(d: Bits) -> Bits:
    """Eq. 1: G <-> B swap on the recipient label, with output inversion."""
    swap = (2, 3, 0, 1, 6, 7, 4, 5)
    return tuple(1 - d[swap[i]] for i in range(8))


def ep_swap(d: Bits) -> Bits:
    """Cooperative <-> Competitive emotion-profile label swap."""
    swap = (1, 0, 3, 2, 5, 4, 7, 6)
    return tuple(d[swap[i]] for i in range(8))


def orbit(d: Bits) -> Set[Bits]:
    """The 4-element symmetry orbit containing ``d``."""
    return {d, paper_mirror(d), ep_swap(d), ep_swap(paper_mirror(d))}


def coop_ones(d: Bits) -> int:
    """Number of 1-outputs (Good reputations) on the cooperative-emotion side.

    O bits are at positions 1, 3, 5, 7 in the flatten order.
    """
    return d[1] + d[3] + d[5] + d[7]


def comp_ones(d: Bits) -> int:
    """Number of 1-outputs (Good reputations) on the competitive-emotion side.

    M bits are at positions 0, 2, 4, 6 in the flatten order.
    """
    return d[0] + d[2] + d[4] + d[6]


def canonical(d: Bits) -> Bits:
    """Canonical representative of the orbit containing ``d``.

    Permissive-on-cooperative convention: maximises
    ``coop_ones(b) - comp_ones(b)`` so the more permissive 2nd-order half-rule
    is displayed on the cooperative-emotion side. Ties broken by the
    Ohtsuki-Iwasa heuristic (prefer CGO=1, CBO=1, DBO=1), then lex.
    """
    def score(b: Bits):
        return (coop_ones(b) - comp_ones(b), b[7], b[5], b[1], b)
    return max(orbit(d), key=score)


def is_trivial(d: Bits) -> bool:
    """True if ``d`` does not discriminate between emotion profiles."""
    return d[0] == d[1] and d[2] == d[3] and d[4] == d[5] and d[6] == d[7]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def canonical_ebsns(include_trivial: bool = False) -> List[List[int]]:
    """Return the canonical set of EBSNs as a list of 8-element ``[0/1]`` lists.

    Parameters
    ----------
    include_trivial : bool, default False
        If False (default), excludes the 16 trivial norms (those whose two
        halves coincide at every action/recipient position, i.e. norms that
        are not actually emotion-based).

    Returns
    -------
    list[list[int]]
        Each inner list has length 8, in the canonical flatten order
        ``(DBM, DBO, DGM, DGO, CBM, CBO, CGM, CGO)``.
    """
    seen: Set[Bits] = set()
    out: List[List[int]] = []
    for i in range(256):
        d: Bits = tuple(int(b) for b in format(i, "08b"))
        if not include_trivial and is_trivial(d):
            continue
        c = canonical(d)
        if c not in seen:
            seen.add(c)
            out.append(list(c))
    return out


def canonical_ebsns_str(include_trivial: bool = False) -> List[str]:
    """Convenience: same as :func:`canonical_ebsns` but returns 8-char strings."""
    return ["".join(str(b) for b in n) for n in canonical_ebsns(include_trivial)]


if __name__ == "__main__":
    norms = canonical_ebsns()
    print(f"Canonical EBSNs (non-trivial, paper Eq.1 symmetry): {len(norms)}")
    for n in norms:
        print("".join(str(b) for b in n))
